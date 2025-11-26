import sys
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from models.ops import pad_tensors_wgrad
from models.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts
from .base_agent import BaseAgent

import numpy as np
import torch
from collections import defaultdict
from contextlib import nullcontext
from models.model import VLNBert
from models.graph_utils import GraphMap
from tasks.datasets.mp3d_dataset import MP3DDataset
from utils.eval_utils import cal_dtw
from typing import List, Dict, Optional

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device
        masks = torch.arange(max_len).to(device).repeat(len(seq_lens), 1) < seq_lens.unsqueeze(1)
        return masks

    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)

    seq_lens = np.array(seq_lens)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


class DUETAgent(BaseAgent):
    name = 'duet'
    
    def __init__(self, args, envs, device=None):
        super().__init__(args, envs, device)

        # setup viewpoint candidates dict
        self.scanvp_cands = {}

    def _build_model(self):
        # NOTE: Because we run tourch.cuda.set_device() in main.py, put model to cuda directly
        # with self.model.cuda() is just fine
        self.model = VLNBert(self.args).to(self.device)
        
    def _language_variable(self, obs):
        ''' Extract language information into variable. '''
        if self.args.model.load_text_features:
            seq_lengths = [len(ob['instr_embedding']) for ob in obs]
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool_)
        for i, ob in enumerate(obs):
            if self.args.model.load_text_features:
                seq_tensor[i, :seq_lengths[i]] = ob['instr_embedding']
            else:
                seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        task = torch.LongTensor([ob['task_id'] for ob in obs]).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask, 'task': task
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.feature.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.feature.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.feature.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.feature.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.agent.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.agent.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def load_balancing_loss_func(
        self, gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        r"""
        Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

        See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
        function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
        experts is too unbalanced.

        Args:
            gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
                Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
                shape [batch_size X sequence_length, num_experts].
            attention_mask (`torch.Tensor`, *optional*):
                The attention_mask used in forward function
                shape [batch_size X sequence_length] if not None.
            num_experts (`int`, *optional*):
                Number of experts

        Returns:
            The auxiliary loss.
        """
        if gate_logits is None or not isinstance(gate_logits, tuple):
            return 0

        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
                router_per_expert_attention_mask, dim=0
            )

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * num_experts


    def _teacher_action(
        self, 
        obs: List[Dict],
        vpids: List[List[str]],
        ended: List[bool],
        task: str,
        dataset: MP3DDataset,
        simulation_envs: List[str],
        visited_masks: Optional[List[List[bool]]] = None,
        imitation_learning: bool = False,
        step: int = 0,
        traj: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        Get teacher action for navigation task.
        """
        a = np.full(len(obs), self.args.agent.ignoreid, dtype=np.int64)

        for i, ob in enumerate(obs):
            # find shortest path and distance according to simulation environment
            shortest_distances = dataset.environments[simulation_envs[i]]['shortest_distances']
            shortest_paths = dataset.environments[simulation_envs[i]]['shortest_paths']
            if ended[i]:
                continue

            if imitation_learning:
                assert ob['viewpoint'] == ob['gt_path'][step]
                a[i] = 0 if step == len(ob['gt_path']) - 1 else vpids[i].index(ob['gt_path'][step + 1])
                continue

            if 'OBJNAV' in task:
                goals_key = ob['goals_key']
                goal_positions = dataset.objects[goals_key][-1]["all_observation_viewpoint_ids"]

                if ob['viewpoint'] == ob['gt_path'][-1] or ob['viewpoint'] in goal_positions:
                    a[i] = 0
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.agent.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and (visited_masks is None or not visited_masks[i][j]):
                            dist = self._calculate_distance(
                                scan, cur_vp, vpid, ob['gt_path'], shortest_distances, shortest_paths, traj[i]['path'], dataset, goal_positions
                            )
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.agent.ignoreid:
                        print(f'scan {scan}: all vps are searched')
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.agent.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and (visited_masks is None or not visited_masks[i][j]):
                            dist = self._calculate_distance(
                                scan, cur_vp, vpid, ob['gt_path'], shortest_distances, shortest_paths, traj[i]['path'], dataset
                            )
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.agent.ignoreid:
                        print(f'scan {scan}: all vps are searched')
        return torch.from_numpy(a).cuda()

    def _calculate_distance(
        self, scan, cur_vp, vpid, gt_path, shortest_distances, shortest_paths, traj_path, dataset, goal_positions=None
    ) -> float:
        if self.args.agent.expert_policy == 'ndtw':
            return -cal_dtw(
                shortest_distances[scan], 
                sum(traj_path, []) + shortest_paths[scan][cur_vp][vpid][1:], 
                gt_path, 
                threshold=3.0
            )['nDTW']
        elif self.args.agent.expert_policy == 'spl':
            # if goal_positions:
            #     nearest_goal_to_end, nav_error = dataset.get_nearest_goal(shortest_distances[scan], goal_positions, [cur_vp])
            #     return nav_error + shortest_distances[scan][cur_vp][vpid]
            return shortest_distances[scan][vpid][gt_path[-1]] + shortest_distances[scan][cur_vp][vpid]
        return float('inf')

    def make_equiv_action(self, a_t, sims, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                sims[i].newEpisode(ob['scan'], action, heading, elevation)

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']
    
    def reset_sims(self, sims, items):
        for sim, item in zip(sims, items):
            scanID = item['scan']
            viewpointID = item['path'][0]
            heading = item['heading']
            sim.newEpisode(scanID, viewpointID, heading)
        
        return sims

    # @profile
    def rollout(
        self,
        input: Dict[str, List],
        dataset: MP3DDataset,
        criterion: torch.nn.Module,
        feedback: str = 'teacher',
        loss_coef: float = None,
        validate: bool = False,
        *args, **kwargs
    ) -> List[Dict]:
        """
        Rollout the agent for one episode.
        Args:
            input: (`Dict[str, List]`)
                The input data for the agent.
            dataset: (`MP3DDataset`)
                The dataset for the agent.
            feedback: (`str`)
                The feedback for the agent. Can be 'teacher', 'argmax', 'sample', 'expl_sample'.
            loss_coef: (`float`)
                The loss coefficient for the agent.
            validate: (`bool`)
                Whether to validate the agent.
        """
        obs = input['observations']
        items = input['items']
        sims = input['sims']
        simulation_envs = [sim.simulation_env for sim in sims]
        data_type = input['data_type'][0].upper()

        max_action_len = self.args.task.val_max_action_len[data_type] if validate else self.args.task.train_max_action_len[data_type]

        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.model('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        entropys = []
        loss = 0.
        nav_loss = 0.
        moe_loss = 0. if self.args.model.use_moe_layer else None

        for t in range(max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.model('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
                'navigation_task': language_inputs['task'],
            })
            nav_outs = self.model('navigation', nav_inputs)

            if self.args.agent.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.agent.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                                        
            if loss_coef is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended, 
                    task=data_type, dataset=dataset, simulation_envs=simulation_envs,
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.agent.fusion != 'local' else None,
                    imitation_learning=(feedback=='teacher'), step=t, traj=traj
                )
                nav_loss += criterion(nav_logits, nav_targets)

                # MOE loss
                if self.args.model.use_moe_layer:
                    all_router_logits = nav_outs['all_router_logits']
                    step_moe_loss = self.load_balancing_loss_func(
                        all_router_logits, self.args.model.num_experts, self.args.model.num_experts_per_tok
                    )
                    step_moe_loss = step_moe_loss.to(nav_loss.device)
                    moe_loss += step_moe_loss
                                                 
            # Determinate the next navigation viewpoint
            if feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.agent.expl_max_ratio  # hyper-param
                if self.args.agent.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if feedback == 'teacher' or feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, sims, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.experiment.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            new_obs = []
            for b_i in range(batch_size):
                new_obs.append(
                    dataset.get_obs(
                        item = items[b_i],
                        sim = sims[b_i],
                        data_type=data_type,
                    )
                )
            obs = new_obs

            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if loss_coef is not None:
            # Calculate the loss
            # the actual batch size is (batch_size * accumulate_gradient_steps * world_size)
            # the loss is averaged over (batch_size * accumulate_gradient_steps)
            # the log is averaged over (batch_size)
            accumulate_gradient_steps = self.args.training.accumulate_gradient_steps
            nav_loss = nav_loss * loss_coef / (batch_size * accumulate_gradient_steps)
            loss += nav_loss
            self.logs['IL_loss'].append(nav_loss.item() * accumulate_gradient_steps)
        
            if moe_loss is not None:
                moe_loss = moe_loss * self.args.model.router_aux_loss_coef / (batch_size * accumulate_gradient_steps)
                loss += moe_loss
                self.logs['moe_loss'].append(moe_loss.item() * accumulate_gradient_steps)
            else:
                self.logs['moe_loss'].append(0)
            
            self.logs['loss'].append(loss.item() * accumulate_gradient_steps)

        return traj, loss