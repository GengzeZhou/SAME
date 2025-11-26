import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict, Union
from logging import Logger
from neptune import Run
from collections import defaultdict
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_constant_schedule_with_warmup
from utils.common_utils import all_gather, merge_dist_results
from tasks.loaders import MetaLoader, PrefetchLoader
from tasks.agents import DUETAgent

class MetaTrainer(type):
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if 'name' in attrs:
            MetaTrainer.registry[attrs['name']] = cls

class BaseTrainer(metaclass=MetaTrainer):
    name = "base"

    def __init__(
        self, 
        config: Dict,
        agent: DUETAgent,
        train_dataloader: PrefetchLoader,
        eval_dataloader: MetaLoader,
        logger: Logger,
        writer: Union[SummaryWriter, Run],
    ):
        self.config = config
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.agent = agent

        # distributed training
        self.rank = config.distributed.rank
        self.local_rank = config.distributed.local_rank
        self.world_size = config.distributed.world_size
        self.distributed = config.distributed.distributed
        self.device = torch.device('cuda', self.local_rank)

        # logging
        self.output_dir = os.path.join(config.experiment.output_dir, config.experiment.id)
        self.writer = writer
        self.use_neptune = config.experiment.use_neptune

        # training
        self.optimizer = torch.optim.AdamW([p for n, p in self.agent.model.named_parameters() if p.requires_grad], lr=self.config.training.learning_rate)
        self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.training.num_warmup_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.agent.ignoreid, reduction='sum')
        self.gradient_clipping = self.config.training.gradient_clipping

        self.total_iterations = self.config.training.iters
        self.num_iters_per_epoch = self.config.training.num_iters_per_epoch
        self.num_epoches = self.total_iterations // self.num_iters_per_epoch
        self.accumulate_gradient_steps = self.config.training.accumulate_gradient_steps if self.config.training.accumulate_gradient_steps is not None else 1

        # resume from checkpoint
        if self.config.experiment.resume_file is not None:
            self.resume_from_checkpoint(config.experiment.resume_file)
        else:
            self.resume_from_epoch = 0
        param_sums = sum(p.numel() for p in self.agent.model.parameters() if p.requires_grad)
        logger.info("Model initialized with {:.2f} M trainable parameters".format(param_sums/1000**2))

        if self.distributed:
            self.agent.model = DDP(self.agent.model, device_ids=[self.local_rank], find_unused_parameters=False)
            logger.info(f"Training with batch size {self.config.training.batch_size} on {self.world_size} GPUs, total batch size {self.config.training.batch_size * self.world_size}.")
        else:
            logger.info(f"Training with batch size {self.config.training.batch_size} on 1 GPU.")
        
        if self.config.training.use_amp:
            self.set_use_amp()
        else:
            self.use_amp = False

    def set_use_amp(self):
        self.use_amp = True
        self.scaler = GradScaler()
    
    def cal_gradient(self, loss):
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def resume_from_checkpoint(self, resume_ckpt) -> int:
        self.logger.info(f"Loading checkpoint from {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location="cpu")
        model_state_dict = self.agent.model.state_dict()
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                self.logger.info(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = self.agent.model.load_state_dict(update_model_state, strict=False)
        self.logger.info(msg)

        if 'epoch' in checkpoint:
            self.resume_from_epoch = checkpoint['epoch'] + 1
            self.total_iterations = checkpoint['total_iterations']
            self.num_iters_per_epoch = checkpoint['num_iters_per_epoch']
            self.num_epoches = self.total_iterations // self.num_iters_per_epoch
            self.logger.info("Resume from Epoch {}".format(self.resume_from_epoch))
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        

    def calc_overall_score(self, results):
        score = 0.
        for task in results:
            if task not in self.config.task.val_source:
                continue
            if task == 'R2R':
                score += results[task]['spl'] / 60
            elif task == 'RXR-EN':
                score += results[task]['spl']
            elif task == 'REVERIE':
                score += results[task]['spl'] / 36.63
            elif task == 'CVDN':
                pass
            elif task == 'SOON':
                score += results[task]['spl'] / 26.58
            elif task == 'OBJNAV_MP3D':
                score += results[task]['spl']
            elif task == 'OBJNAV_HM3D':
                score += results[task]['spl']
            else:
                raise NotImplementedError(f"The method for calculating the score of {task} is not Implemented.")

        return score

    def test(self):
        self.logger.info("**************************** Test ****************************")
        results = self.val_one_epoch(epoch=0)


    def train(self):
        self.logger.info("**************************** Train ****************************")

        best_results, best_score = None, None
        history_scores = []

        for epoch in range(self.resume_from_epoch, self.num_epoches):
            # training
            self.train_one_epoch(epoch)

            # evaluation
            results = self.val_one_epoch(epoch + 1)

            if self.rank==0:
                score = self.calc_overall_score(results)
                history_scores.append(score)
                should_save_checkpoint = False

                if best_results is None or score > best_score:
                    best_results = results
                    best_score = score
                    should_save_checkpoint = self.config.experiment.max_saved_ckpts > 0
                
                self.logger.info(f"Current Score: {score}")
                self.logger.info(f"Best Score: {best_score}")

                # Save the best
                if should_save_checkpoint:
                    if len(history_scores) > self.config.experiment.max_saved_ckpts:
                        sorted_scores = sorted(enumerate(history_scores), key=lambda x: x[1], reverse=True)
                        
                        remove_epoch = sorted_scores[self.config.experiment.max_saved_ckpts][0] + 1
                        remove_model_path = os.path.join(self.output_dir, f"ckpts/epoch_{remove_epoch}.pt")

                        if os.path.exists(remove_model_path):
                            os.remove(remove_model_path)
                            self.logger.info(f"Remove Checkpoint at Epoch {remove_epoch}...")

                    model_path = os.path.join(self.output_dir, f"ckpts/epoch_{epoch + 1}.pt")
                    self.save_checkpoint(model_path, epoch, self.total_iterations, self.num_iters_per_epoch)

            if self.config.experiment.save_latest_states:
                model_path = os.path.join(self.output_dir, f"ckpts/latest.pt")
                self.save_checkpoint(model_path, epoch, self.total_iterations, self.num_iters_per_epoch)
        
        # print best results
        if self.rank == 0:
            self.logger.info(f"Best Results:")
            self.logger.info(best_results)


    def train_one_epoch(
        self,
        epoch: int,
    ):  
        # set up logger
        self.agent.logs = defaultdict(list)
        loss_stats = {k: list() for k in self.config.task.source}

        self.agent.model.train()

        # Train for num_iters_per_epoch iterations for each epoch or
        # Do a full iteration over the training set

        pbar = tqdm(
            range(self.num_iters_per_epoch),
            disable=self.rank!=0,
            total=self.total_iterations,
            initial=(epoch * self.num_iters_per_epoch),
        )

        for step, (name, batch) in enumerate(self.train_dataloader):
            task_name, split = name.split(".")
            task_loss_coef = self.config.task.loss_coef.get(task_name, 1.)
            IL_loss_coef = self.config.training.IL_loss_coef
            dataset = self.train_dataloader.loader.get_dataset(name)

            # Set supervision
            if self.config.training.train_alg == 'imitation':
                # imitation learning
                feedback = "teacher"
                traj, loss = self.agent.rollout(
                    input=batch,
                    dataset=dataset,
                    criterion=self.criterion,
                    feedback=feedback,
                    loss_coef=task_loss_coef * IL_loss_coef,
                    validate=False,
                )
            elif self.config.training.train_alg == 'dagger':
                # DAgger
                if IL_loss_coef != 0:
                    # Do an imitation learning step but not logging the results
                    feedback = "teacher"
                    traj, IL_loss = self.agent.rollout(
                        input=batch,
                        dataset=dataset,
                        criterion=self.criterion,
                        feedback=feedback,
                        loss_coef=task_loss_coef * IL_loss_coef,
                        validate=False,
                    )
                    # Reset the simulator states
                    self.agent.reset_sims(batch['sims'], batch['items'])
                    # Calculate the imitation learning loss
                    if not self.config.training.accumulate_IL_grad:
                        self.cal_gradient(IL_loss)
                # Do a student forcing step
                feedback = 'expl_sample' if self.config.agent.expl_sample else 'sample'
                traj, loss = self.agent.rollout(
                    input=batch,
                    dataset=dataset,
                    criterion=self.criterion,
                    feedback=feedback,
                    loss_coef=task_loss_coef,
                    validate=False,
                )
                loss = loss + IL_loss

            loss_stats[task_name].append(self.agent.logs['loss'][-1])

            self.cal_gradient(loss)

            if (step+1) % self.accumulate_gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.gradient_clipping)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            self.lr_scheduler.step()

            if self.rank == 0:
                avg_loss = sum(self.agent.logs['loss']) / max(len(self.agent.logs['loss']), 1)
                avg_IL_loss = sum(self.agent.logs['IL_loss']) / max(len(self.agent.logs['IL_loss']), 1)
                avg_moe_loss = sum(self.agent.logs['moe_loss']) / max(len(self.agent.logs['moe_loss']), 1)
                if self.config.training.train_alg == 'dagger':
                    avg_entropy = sum(self.agent.logs['entropy']) / max(len(self.agent.logs['entropy']), 1)
                else:
                    avg_entropy = 0
                if not self.use_neptune:
                    # tensorboard
                    self.writer.add_scalar("train/loss", self.agent.logs['loss'][-1], epoch * self.num_iters_per_epoch + step)
                    self.writer.add_scalar("train/IL_loss", self.agent.logs['IL_loss'][-1], epoch * self.num_iters_per_epoch + step)
                    self.writer.add_scalar("train/moe_loss", self.agent.logs['moe_loss'][-1], epoch * self.num_iters_per_epoch + step)
                    self.writer.add_scalar("train/avg_loss", avg_loss, epoch * self.num_iters_per_epoch + step)
                    self.writer.add_scalar("train/avg_IL_loss", avg_IL_loss, epoch * self.num_iters_per_epoch + step)
                    self.writer.add_scalar("train/avg_moe_loss", avg_moe_loss, epoch * self.num_iters_per_epoch + step)
                    if self.config.training.train_alg == 'dagger':
                        self.writer.add_scalar("train/entropy", self.agent.logs['entropy'][-1], epoch * self.num_iters_per_epoch + step)
                        self.writer.add_scalar("train/avg_entropy", avg_entropy, epoch * self.num_iters_per_epoch + step)
                    for k in self.config.task.source:
                        self.writer.add_scalar(f"train/tasks/{k}_loss", sum(loss_stats[k]) / max(len(loss_stats[k]), 1), epoch * self.num_iters_per_epoch + step)
                else:
                    # neptune
                    self.writer['train/loss'].append(value=self.agent.logs['loss'][-1], step=epoch * self.num_iters_per_epoch + step)
                    self.writer['train/IL_loss'].append(value=self.agent.logs['IL_loss'][-1], step=epoch * self.num_iters_per_epoch + step)
                    self.writer['train/moe_loss'].append(value=self.agent.logs['moe_loss'][-1], step=epoch * self.num_iters_per_epoch + step)
                    self.writer['train/avg_loss'].append(value=avg_loss, step=epoch * self.num_iters_per_epoch + step)
                    self.writer['train/avg_IL_loss'].append(value=avg_IL_loss, step=epoch * self.num_iters_per_epoch + step)
                    self.writer['train/avg_moe_loss'].append(value=avg_moe_loss, step=epoch * self.num_iters_per_epoch + step)
                    if self.config.training.train_alg == 'dagger':
                        self.writer['train/entropy'].append(value=self.agent.logs['entropy'][-1], step=epoch * self.num_iters_per_epoch + step)
                        self.writer['train/avg_entropy'].append(value=avg_entropy, step=epoch * self.num_iters_per_epoch + step)
                    for k in self.config.task.source:
                        self.writer[f"train/tasks/{k}_loss"].append(value=sum(loss_stats[k]) / max(len(loss_stats[k]), 1), step=epoch * self.num_iters_per_epoch + step)

                # Log the training stats
                verbose_dict = dict(
                    step=step,
                    name=name,
                    loss=avg_loss,
                    entropy=avg_entropy,
                    lr=self.lr_scheduler.get_last_lr()[0],
                )
                for k in self.config.task.source:
                    verbose_dict[k] = sum(loss_stats[k]) / max(len(loss_stats[k]), 1)
                pbar.set_postfix(verbose_dict)
                pbar.update()

            if step == self.num_iters_per_epoch - 1:
                self.logger.info(f"Validation after training for {epoch * self.num_iters_per_epoch + step + 1} iterations, epoch {epoch + 1}")
                # train_stat_str = 'Loss: %.2f\n' % avg_loss
                # for task in self.config.task.source:
                #     train_stat_str += "%s: %.2f\n" % (task, sum(loss_stats[task]) / max(len(loss_stats[task]), 1))
                # pretty print the training stats
                train_stat_str = f"Training loss at epoch {epoch + 1}: {avg_loss:.2f}"
                for task in self.config.task.source:
                    train_stat_str += f"\nDataset: {task}, Loss: {sum(loss_stats[task]) / max(len(loss_stats[task]), 1):.2f}"
                break
        
        self.logger.info(train_stat_str)


    @torch.no_grad()
    def val_one_epoch(
        self,
        epoch: int,
    ) -> Dict[str, Dict[str, float]]:

        self.agent.model.eval()

        loss_str = f"Evaluation at epoch {epoch}, {epoch * self.num_iters_per_epoch} training iterations:"
        task_results = {}
        for name, loader in self.eval_dataloader.items():
            task_name, split = name.split(".")
            self.logger.info(f"Validating the {task_name} task on the {split} split.")
            dataset = self.eval_dataloader[name].get_dataset()
            looped = False
            preds = []
            results = {}
            loader = tqdm(loader, disable=self.rank!=0)
            for i, batch in enumerate(loader):
                traj, loss = self.agent.rollout(
                    input=batch,
                    dataset=dataset,
                    criterion=self.criterion,
                    feedback="argmax",
                    validate=True,
                )

                for s_traj in traj:
                    if s_traj['instr_id'] in results:
                        looped = True
                    else:
                        results[s_traj['instr_id']] = s_traj
                    
                if looped:
                    break

            preds = self.agent.get_results(results)

            all_preds = all_gather(preds)
            all_preds = merge_dist_results(all_preds)

            item_metrics = None
            if self.rank == 0 and not split.startswith('test'):
                score_summary, item_metrics = dataset.eval_metrics(all_preds, logger=self.logger, name=task_name)

                task_results[task_name] = score_summary
                # loss_str += f"\n\033[1;32m[Eval]\033[0m Dataset: \033[1;94m{task_name}\033[0m, Split: \033[1;94m{split}\033[0m"
                loss_str += f"\nDataset: {task_name}, Split: {split}"
                for metric, val in score_summary.items():
                    # Log the evaluation stats
                    if not self.use_neptune:
                        self.writer.add_scalar(f"{task_name}/{split}_{metric}", val, epoch * self.num_iters_per_epoch)
                    else:
                        self.writer[f"eval/{task_name}/{split}_{metric}"].append(value=val, step=epoch * self.num_iters_per_epoch)
                    # Print the evaluation stats
                    if metric == 'sr':
                        loss_str += '\n\033[43m%s: %.2f\033[0m' % (metric, val)
                    else:
                        loss_str += ', %s: %.2f' % (metric, val)
            
            if self.rank == 0:
                dataset.save_json(
                    all_preds, 
                    os.path.join(self.output_dir, "results", f"{task_name}_{split}_results.json"),
                    item_metrics=item_metrics if item_metrics is not None else None
                )

        self.logger.info(loss_str)
        
        return task_results

    def save_checkpoint(
        self, 
        model_path: str,
        epoch: int=0, 
        total_iterations: int=500000,
        num_iters_per_epoch: int=5000,
        save_optimizer_states: bool=True
    ):  
        model = self.agent.model
        if hasattr(model, 'module'):
            model = model.module
        
        state_dict = {
            "model_state_dict": model.state_dict()
        }
        if save_optimizer_states:
            state_dict.update({
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "total_iterations": total_iterations,
                "num_iters_per_epoch": num_iters_per_epoch
            })

        torch.save(state_dict, model_path)