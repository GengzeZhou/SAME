import os
import json
import copy
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from omegaconf import DictConfig
from logging import Logger
from typing import List, Dict, Tuple, Union, Iterator

from tasks.datasets.mp3d_envs import load_graphs
from tasks.datasets import load_dataset
from .feature_db import create_feature_db, create_object_feature_db



def build_dataloader(dataset, distributed, training, batch_size, num_workers):
    if distributed:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=training
        )
        pre_epoch = sampler.set_epoch
    else:
        # not distributed
        if training:
            sampler: Union[
                RandomSampler, SequentialSampler, DistributedSampler
            ] = RandomSampler(dataset)
            # sampler = SequentialSampler(dataset)  # Debug Mode
        else:
            sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        # if size > 1:
        #     batch_size *= size

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_batch,
    )
    loader.num_batches = len(loader)

    return loader, pre_epoch


def move_to_cuda(batch: Union[List, Tuple, Dict, torch.Tensor], device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        return [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t, device) for n, t in batch.items()}
    return batch

class MetaLoader:
    """wraps multiple data loaders"""

    def __init__(
        self, loaders, accum_steps: int = 1, distributed: bool = False, device=None, off_batch_task: bool = False,
    ):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.name2pre_epoch = {}
        self.names: List[str] = []
        ratios: List[int] = []

        self.num_batches = 0
        self.off_batch_task = off_batch_task

        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r, p = l
            elif isinstance(l, DataLoader):
                r = 1
                p = lambda e: None
            else:
                raise ValueError()
            self.names.append(n)
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2pre_epoch[n] = p
            ratios.append(r)

            self.num_batches += l.num_batches

        self.accum_steps = accum_steps
        self.device = device
        self.sampling_ratios = torch.tensor(ratios).float().to(self.device)
        self.distributed = distributed
        self.step = 0
        self.epoch_id = 0  

    def get_dataset(self, name):
        return self.name2loader[name].dataset

    def __iter__(self) -> Iterator[Tuple]:
        """this iterator will run indefinitely"""
        task_id = None
        self.step = 0
        while True:
            # if self.step % self.accum_steps == 0:
            task_id = torch.multinomial(self.sampling_ratios, 1)
            if self.distributed and not self.off_batch_task:
                # make sure all process is training same task
                dist.broadcast(task_id, 0)
            self.step += 1
            task = self.names[task_id.cpu().item()]
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:

                self.epoch_id += 1
                # In distributed mode, calling the set_epoch() method at the beginning of each epoch
                # before creating the DataLoader iterator is necessary to make shuffling work properly
                # across multiple epochs. Otherwise, the same ordering will be always used.
                self.name2pre_epoch[task](self.epoch_id)
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    """
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.num_batches = self.loader.num_batches

    def get_dataset(self):
        return self.loader.dataset

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        batch = self.batch
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def create_environments(
        config: DictConfig,
        logger: Logger,
    ) -> Dict[str, object]:
    """
    Create environments for training and validation.
    """
    logger.info("Loading simulation environments...")

    task_cfg = copy.deepcopy(config.task)
    simulator_cfg = copy.deepcopy(config.simulator)

    # Simulation environment
    simulator_list = []
    for task_name in task_cfg.source:
        if task_name in task_cfg.train_simulation_env or task_name in task_cfg.eval_simulation_env:
            simulation_envs =  task_cfg.train_simulation_env[task_name]
            if isinstance(simulation_envs, str):
                simulation_envs = [simulation_envs]
            simulator_list += simulation_envs
    simulator_list = list(set(simulator_list))

    environments = {}
    for simulator_name in simulator_list:
        if config.experiment.debug and simulator_name == "hm3d_habitat":
            continue
        if config.experiment.test and simulator_name == "hm3d_habitat":
            continue
        connectivity_dir = simulator_cfg.connectivity_dir[simulator_name]
        candidate_file_dir = simulator_cfg.candidate_file_dir[simulator_name]
        node_location_dir = simulator_cfg.node_location_dir[simulator_name]

        with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
            scans = [x.strip() for x in f]
        
        # Load connectivity graph for each scan
        graphs, shortest_paths, shortest_distances = load_graphs(connectivity_dir, scans)
        candidate_dict = json.load(open(candidate_file_dir, 'r'))

        logger.info(f"{simulator_name}: loaded {len(scans)} scans")

        environments[simulator_name] = {
            'graphs': graphs,
            'shortest_paths': shortest_paths,
            'shortest_distances': shortest_distances,
            'candidate_dict': candidate_dict,
            'node_location_dir': node_location_dir
        }
    
    return environments


def create_dataloaders(
    config: DictConfig,
    logger: Logger,
    environments: Dict[str, object],
    training: bool,
    device: torch.device,
) -> Union[Tuple[Dict[str, PrefetchLoader], Dict[str, object]], Tuple[PrefetchLoader, Dict[str, object]]]:
    """
    Create data loaders for training and validation.

    Args:
        config (`DictConfig`):
            Configurations for the experiment.
        logger (`Logger`):
            Logger object.
        training (`bool`):
            Whether to create training or validation data loaders.
        device (`torch.device`):
            Device to move data to.
    """
    task_cfg = copy.deepcopy(config.task)
    training_cfg = copy.deepcopy(config.training)


    # Create feature database
    feat_db = create_feature_db(config)
    obj_feat_db = create_object_feature_db(config)
    
    dataloaders = {}

    # load datasets
    if config.experiment.test:
        dataset_list = task_cfg.test_source
    elif task_cfg.val_source is not None and not training:
        dataset_list = task_cfg.val_source
    else:
        dataset_list = copy.deepcopy(task_cfg.source)
    
    for k, task_name in enumerate(dataset_list):
        if training:
            splits = ['train']
        elif config.experiment.test:
            splits = ['test']
        else:
            splits = task_cfg.eval_splits[task_name]
            if isinstance(splits, str):
                splits = [splits]
        
        # load dataset by names
        for split in splits:
            dataset = load_dataset(
                name=task_name.lower(), 
                config=config, 
                split=split, 
                environments=environments, 
                logger=logger, 
                source=task_name
            )

            # assign feature database
            task_feat_db = {}
            for simulator_name in dataset.simulation_envs:
                task_feat_db[simulator_name] = feat_db[simulator_name]
            
            # assign object database
            if config.feature.enable_og:
                if task_name in ["REVERIE", "REVERIE_AUG"]:
                    task_obj_feat_db = obj_feat_db['reverie']
                elif task_name == "SOON":
                    task_obj_feat_db = obj_feat_db['soon']
                else:
                    task_obj_feat_db = None
            else:
                task_obj_feat_db = None

            dataset.init_feat_db(feat_db=task_feat_db, obj_feat_db=task_obj_feat_db)

            task_loader, pre_epoch = build_dataloader(
                dataset, distributed=config.distributed.distributed,
                training=training, batch_size=training_cfg.batch_size if training else training_cfg.val_batch_size, num_workers=training_cfg.workers
            )

            dataloader_name = f"{task_name}.{split}"
            if training:
                ratio = task_cfg.ratio[k]
                dataloaders[dataloader_name] = (task_loader, ratio, pre_epoch)
            else:
                dataloaders[dataloader_name] = PrefetchLoader(task_loader, device=device)

    if training:
        metaloader = MetaLoader(
            dataloaders,
            accum_steps=training_cfg.accumulate_gradient_steps,
            distributed=config.distributed.distributed,
            device=device,
            off_batch_task=training_cfg.off_batch_task
        )
        metaloader = PrefetchLoader(metaloader, device)

        if training_cfg.num_iters_per_epoch!=-1:
            metaloader.num_batches = training_cfg.num_iters_per_epoch
        return metaloader
    else:
        return dataloaders