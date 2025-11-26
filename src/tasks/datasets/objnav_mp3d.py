import os
import json
import numpy as np
from tqdm import tqdm
from logging import Logger
from omegaconf import DictConfig
from typing import List, Dict
from .mp3d_dataset import MP3DDataset
from utils.eval_utils import cal_dtw, cal_cls
from collections import defaultdict
ERROR_MARGIN = 3.0

class ObjNavMP3DDataset(MP3DDataset):
    name = "objnav_mp3d"

    def __init__(
        self,
        config: DictConfig,
        split: str,
        environments: Dict,
        logger: Logger = None,
        source: str = None,
    ):
        self.shortest_paths = environments['mp3d_habitat']["shortest_paths"]
        super().__init__(config, split, environments, logger, source)
        

    def load_data(self, anno_file, max_instr_len=200, debug=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """

        data = []
        scenes = os.listdir(anno_file)
        for scene in scenes:
            scene_path = os.path.join(anno_file, scene)
            with open(scene_path, 'r') as f:
                new_data = json.load(f)
            # Join
            data.append(new_data)

        new_data = []
        self.objects = {}

        data = tqdm(data, desc="Loading data")
        for i, item in enumerate(data):
            scene_objects = item[0]
            episodes = item[1:]
            for episode in episodes:
                episode_id = episode['episode_id']
                scene_id = episode['scene_id'].split('/')[1]
                goals_key = episode['goals_key']

                # for 'mp3d_val' or 'val_train_seen' in habitat-web, don't need to refine the path
                if 'val' not in self.split:
                    if len(episode['path']) > 25:
                        goal_locations = scene_objects[goals_key][-1]["all_observation_viewpoint_ids"]
                        goal_viewpoint_id, goal_path = self.get_goal(self.shortest_paths[scene_id], goal_locations, episode['path'][0])
                        # replace the path with the goal path
                        if goal_path:
                            episode['path'] = goal_path
                    

                    del episode['original_path']
                    del episode['refined_path']

                episode['scan'] = scene_id
                episode['instr_id'] = f'objnav_mp3d_{scene_id}_{episode_id}'
                episode['path_id'] = episode_id
                episode['sample_idx'] = i
                episode['raw_idx'] = i
                episode["heading"] = 0
                episode['data_type'] = 'objnav_mp3d'

                new_data.append(episode)
            self.objects.update(scene_objects)

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['goals_key'], x['path']) \
                for x in new_data if 'path' in x
        }
        return new_data, gt_trajs

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
            'goals_key': item['goals_key'],
        }
        
        if self.obj_feat_db is not None:
            obj_info = self.get_object_info(item, state)
            ob.update(obj_info)
            ob['distance'] = 0
        else:
            ob['distance'] = 0
            
        return ob

    def eval_metrics(self, preds, logger, name):
        """
        Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]
        :param preds:
        :param logger:
        :param name:
        :return:
        """
        logger.info('Evaluated %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, goals_key, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self.eval_dis_item(scan, goals_key, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }

        return avg_metrics, metrics

    def eval_dis_item(self, scan, goals_key, pred_path, gt_path):
        scores = {}

        shortest_distances = self.environments['mp3d_habitat']["shortest_distances"][scan]
        goal_positions = self.objects[goals_key][-1]["all_observation_viewpoint_ids"]
        goal_positions.append(gt_path[-1])

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position, nearest_goal = self.get_nearest(shortest_distances, goal_positions, path)
        nearest_goal_to_end, nav_error = self.get_nearest_goal(shortest_distances, goal_positions, path)

        scores['nav_error'] = nav_error
        scores['oracle_error'] = shortest_distances[nearest_position][nearest_goal]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def get_nearest(self, shortest_distances, goals_id, path):
        near_id = path[0]
        near_d = np.inf
        for goal in goals_id:
            for item in path:
                try:
                    d = shortest_distances[item][goal]
                except KeyError:
                    # Goal not reachable from this viewpoint
                    continue
                if d < near_d:
                    near_id = item
                    goal_id = goal
                    near_d = d
        return near_id, goal_id

    def get_nearest_goal(self, shortest_distances, goals_id, path):
        near_d = np.inf
        for goal in goals_id:
            try:
                d = shortest_distances[path[-1]][goal]
            except KeyError:
                # Goal not reachable from this viewpoint
                continue

            if d < near_d:
                goal_id = goal
                near_d = d
        return goal_id, near_d
    
    def get_goal(self, shortest_paths, goals_id, start_vp):
        goal_id, path = None, None
        for goal in goals_id:
            try:
                path = shortest_paths[start_vp][goal]
            except KeyError:
                # Goal not reachable from this viewpoint
                continue

            if 2 < len(path) < 15:
                goal_id = goal
                break
        return goal_id, path

    def save_json(self, results, path, item_metrics=None):
        if item_metrics is not None:
            for k in item_metrics:
                for item, v in zip(results, item_metrics[k]):
                    item[k] = v

        for item in results:
            item['instr_id'] = "_".join(item['instr_id'].split("_")[1:])
            item['trajectory'] = [[y, 0, 0] for x in item['trajectory'] for y in x]

        with open(path, 'w') as fout:
            json.dump(results, fout)