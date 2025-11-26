import os
import copy
import math
import torch
import numpy as np
from logging import Logger
from omegaconf import DictConfig
from typing import List, Dict, Tuple, Union
from collections import defaultdict
from .base_dataset import BaseDataset
from .mp3d_envs import (
    Simulator,
    angle_feature,
    get_all_point_angle_feature,
)


class MP3DDataset(BaseDataset):
    TASK_ID = {
        'R2R': 0,
        'R2R_PREVALENT': 0,
        'R2R_SCALEVLN': 0,
        'REVERIE': 1,
        'REVERIE_SCALEVLN': 1,
        'SOON': 2,
        'RXR-EN': 3,
        'CVDN': 4,
        'OBJNAV_MP3D': 5,
    }

    def __init__(
        self,
        config: DictConfig,
        split: str,
        environments: Dict,
        logger: Logger = None,
        source: str = None,
    ):
        """
        MP3D Dataset for VLN tasks (R2R, REVERIE, CVDN, SOON, EQA)

        Args:
            config (`DictConfig`):
                Experiment configuration.
            split (`str`):
                Dataset split.
            training (`bool`):
                Whether to create training or validation data loaders.
            logger (`Logger`):
                Logger object.
            source (`str`):
                Dataset source, e.g., R2R, REVERIE, CVDN, SOON, EQA.
            nav_graphs (`Dict`):
                Navigation graphs.
                It is a dictionary containing graphs, shortest paths, and shortest distances.
        """
        super().__init__()
        self.config = config
        self.angle_feat_size = self.config.feature.angle_feat_size
        self.logger = logger
        self.debug = config.experiment.debug
        self.source = source
        self.split = split

        if self.split == "train":
            self.training = True
            self.simulation_envs = config.task.train_simulation_env[source]
            if isinstance(self.simulation_envs, str):
                self.simulation_envs = [self.simulation_envs]
            self.max_objects = self.config.model.max_objects
            self.multi_endpoints = True
        else:
            self.training = False
            self.simulation_envs = config.task.eval_simulation_env[source]
            if isinstance(self.simulation_envs, str):
                self.simulation_envs = [self.simulation_envs]
            self.max_objects = None
            self.multi_endpoints = False

        self.batch_size = config.training.batch_size
        self.seed = config.experiment.seed
        self.feat_db = None
        self.obj_feat_db = None

        # load mp3d dataset
        msg = self._load_data(config.dataset, config.experiment.data_dir)   # load data, self.scans is initialized here
        self.buffered_state_dict = {}

        # simulator
        self.environments = {}
        for simulator_name in self.simulation_envs:
            if simulator_name not in environments:
                raise ValueError(f"Simulator {simulator_name} not found in the simulator configuration.")
            all_graphs = environments[simulator_name]['graphs']
            all_shortest_paths = environments[simulator_name]['shortest_paths']
            all_shortest_distances = environments[simulator_name]['shortest_distances']

            self.environments[simulator_name] = {
                'graphs': {scan: all_graphs[scan] for scan in self.scans},
                'shortest_paths': {scan: all_shortest_paths[scan] for scan in self.scans},
                'shortest_distances': {scan: all_shortest_distances[scan] for scan in self.scans},
                'candidate_dict': environments[simulator_name]['candidate_dict'],
                'node_location_dir': environments[simulator_name]['node_location_dir'],
            }

        # angle features
        self.angle_feature = get_all_point_angle_feature(self.angle_feat_size)

        if logger is not None:
            logger.info(f"{source}: {self.__class__.__name__} loaded {len(self.alldata)} samples from {self.split} split")
            logger.info(msg)
        del self.data

    def init_feat_db(self, feat_db, obj_feat_db=None):
        self.feat_db = feat_db
        self.obj_feat_db = obj_feat_db

    def _load_data(self, config, data_dir):
        self.data = dict()
        self.alldata = []
        msg = ""
        if self.source == "R2R":
            anno_file = os.path.join(data_dir, config.R2R.DIR, config.R2R.SPLIT[self.split])
            self.logger.info(f"Loading R2R data from {anno_file}")
            self.data['r2r'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} R2R samples'.format(len(self.data['r2r']))
        elif self.source == "REVERIE":
            anno_file = os.path.join(data_dir, config.REVERIE.DIR, config.REVERIE.SPLIT[self.split])
            bbox_file = os.path.join(data_dir, config.REVERIE.DIR, config.REVERIE.bbox_file)
            obj2vps = self.load_obj2vps(bbox_file)
            self.logger.info(f"Loading REVERIE data from {anno_file}")
            self.data['reverie'], self.gt_trajs = self.load_data(anno_file=anno_file, obj2vps=obj2vps, debug=self.debug)
            msg += '\n- Dataset: load {} REVERIE samples'.format(len(self.data['reverie']))
        elif self.source == "CVDN":
            anno_file = os.path.join(data_dir, config.CVDN.DIR, config.CVDN.SPLIT[self.split])
            self.logger.info(f"Loading REVERIE data from {anno_file}")
            self.data['cvdn'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} CVDN samples'.format(len(self.data['cvdn']))
        elif self.source == "SOON":
            anno_file = os.path.join(data_dir, config.SOON.DIR, config.SOON.SPLIT[self.split])
            self.logger.info(f"Loading SOON data from {anno_file}")
            self.data['soon'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} SOON samples'.format(len(self.data['soon']))
        elif self.source == "RXR-EN":
            anno_file = os.path.join(data_dir, config.RXR_EN.DIR, config.RXR_EN.SPLIT[self.split])
            self.logger.info(f"Loading RXR-EN data from {anno_file}")
            self.data['rxr-en'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} RXR-EN samples'.format(len(self.data['rxr-en']))
        elif self.source == "R2R_SCALEVLN":
            anno_file = os.path.join(data_dir, config.R2R_SCALEVLN.DIR, config.R2R_SCALEVLN.SPLIT[self.split])
            self.logger.info(f"Loading R2R_SCALEVLN data from {anno_file}")
            self.data["r2r_scalevln"], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} R2R_SCALEVLN samples'.format(len(self.data["r2r_scalevln"]))
        elif self.source == "R2R_PREVALENT":
            anno_file = os.path.join(data_dir, config.R2R_PREVALENT.DIR, config.R2R_PREVALENT.SPLIT[self.split])
            self.logger.info(f"Loading R2R_PREVALENT data from {anno_file}")
            self.data["r2r_prevalent"], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} R2R_PREVALENT samples'.format(len(self.data["r2r_prevalent"]))
        elif self.source == "REVERIE_SCALEVLN":
            anno_file = os.path.join(data_dir, config.REVERIE_SCALEVLN.DIR, config.REVERIE_SCALEVLN.SPLIT[self.split])
            bbox_file = os.path.join(data_dir, config.REVERIE_SCALEVLN.DIR, config.REVERIE_SCALEVLN.bbox_file)
            obj2vps = self.load_obj2vps(bbox_file)
            self.logger.info(f"Loading REVERIE_SCALEVLN data from {anno_file}")
            self.data["reverie_scalevln"], self.gt_trajs = self.load_data(anno_file=anno_file, obj2vps=obj2vps, debug=self.debug)
            msg += '\n- Dataset: load {} REVERIE_SCALEVLN samples'.format(len(self.data["reverie_scalevln"]))
        elif self.source == "OBJNAV_MP3D":
            anno_file = os.path.join(data_dir, config.OBJNAV_MP3D.DIR, config.OBJNAV_MP3D.SPLIT[self.split])
            self.logger.info(f"Loading MP3D data from {anno_file}")
            self.data["objnav_mp3d"], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} MP3D samples'.format(len(self.data["objnav_mp3d"]))
        else:
            print("Dataset Source: {}".format(self.source))
            raise NotImplementedError

        for key, value in self.data.items():
            self.alldata += value

        msg += '\n- Dataset: load {} split: {} samples in total'.format(self.split, len(self.alldata))
        self.scans = set([x['scan'] for x in self.alldata])
        msg += '\n- Dataset: load {} split: {} scans in total'.format(self.split, len(self.scans))

        return msg


    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, index):
        item = copy.deepcopy(self.alldata[index])
        item = self.preprocess_item(item)
        data_type = item['data_type']
        scan = item['scan']
        instr_id = item['instr_id']

        # Load the environment
        # If there is only one simulator, use it
        if len(self.simulation_envs) == 1:
            simulator_name = self.simulation_envs[0]
        # Some path could not be transformed to the habitat path, use original simulator
        elif "habitat_path" not in item:
            simulator_name = "mattersim"
        # If there are multiple environments, randomly select one
        else:
            simulator_name = np.random.choice(self.simulation_envs)

        scanId = scan
        heading = item['heading']
        # If select habitat from multiple environments, use the habitat path
        if len(self.simulation_envs) > 1 and simulator_name == "mp3d_habitat":
            viewpointId = item['habitat_path'][0]
            item['path'] = item['habitat_path']
        # Otherwise, use the original path
        else:
            viewpointId = item['path'][0]

        sim = Simulator(
            node_location_dir=self.environments[simulator_name]['node_location_dir'], 
            simulation_env=simulator_name
        )
        sim.newEpisode(scanId, viewpointId, heading)
        observation = self.get_obs(item=item, sim=sim, data_type=data_type)

        # Return initial observations for the trajectory
        data_dict = {
            'data_type': data_type,
            'sample_idx': index,
            'instr_id': instr_id,
            'observations': observation,
            'sims': sim,
            'items': item,
        }

        return data_dict
    
    def preprocess_item(self, item):
        return item

    @staticmethod
    def collate_batch(
        batch_list: List[Dict],
        _unused: bool = False,
    ) -> Dict:
        # batch list is a list of dictionaries from __getitem__
        data_dict = defaultdict(list)

        # collate the data dictionaries from the batch list into a single dictionary
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['NotImplemented']:
                    ret[key] = torch.stack(val, 0)
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
    
    def get_object_info(self, item):
        raise NotImplementedError

    def get_obs(self, item, sim, data_type=None):
        state = sim.getState()

        base_view_id = state["viewIndex"]
        simulation_env = state["simulation_env"]

        feature = self.feat_db[simulation_env].get_image_feature(state["scanId"], state["viewpointId"])

        # Full features
        candidate = self.make_candidate(feature, state["scanId"], state["viewpointId"], state["viewIndex"], simulation_env)

        # [visual_feature, angle_feature] for views
        feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

        ob = {
            'task_id': self.TASK_ID[data_type.upper()],
            'instr_id': item['instr_id'],
            'scan': state["scanId"],
            'viewpoint': state["viewpointId"],
            'viewIndex': state["viewIndex"],
            'position': (state["x"], state["y"], state["z"]),
            'heading': state["heading"],
            'elevation': state["elevation"],
            'feature': feature,
            'candidate': candidate,
            'instruction': item.get('instruction', None),
            'instr_encoding': item['instr_encoding'],
            'gt_path': item['path'],
            'path_id': item['path_id'],
        }
        if self.obj_feat_db is not None:
            obj_info = self.get_object_info(item, state)
            ob.update(obj_info)
            ob['distance'] = 0
        else:
            ob['distance'] = 0
            
        return ob

    def make_candidate(self, feature, scanId, viewpointId, viewId, simulation_env):
        '''
        Make candidate list for the current state.
        The candidate list is formed from the pre-computed candidate dictionary:
        {
            'long_id': {
                'viewpointId': [pointId, distance, heading, elevation, position],
                ...
        }
        '''

        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        long_id = "%s_%s" % (scanId, viewpointId)

        candidate = self.environments[simulation_env]['candidate_dict'][long_id]
        candidate_new = []
        for key, value in candidate.items():
            loc_heading = value[2] - base_heading
            loc_elevation = value[3] - base_elevation
            angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
            visual_feat = feature[value[0]]

            c_new = {
                'heading' : loc_heading,
                'elevation' : loc_elevation,
                'normalized_heading': value[2],
                'normalized_elevation': value[3],
                'scanId': scanId,
                'viewpointId': key,
                'pointId': value[0],
                'distance': value[1],
                'feature': np.concatenate((visual_feat, angle_feat), -1),
                'position': tuple(value[4]),
            }
            candidate_new.append(c_new)
        return candidate_new

    def get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id