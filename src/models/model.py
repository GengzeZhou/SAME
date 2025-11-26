import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vlnbert_init import get_vlnbert_models

class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.model.feat_dropout)
        
    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':            
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s'%mode)