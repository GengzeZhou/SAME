import json
import numpy as np
from tqdm import tqdm
from .reverie import REVERIEDataset

class REVERIEScaleVLNDataset(REVERIEDataset):
    name = "reverie_scalevln"

    def load_data(self, anno_file, obj2vps, debug=False):
        if str(anno_file).endswith("json"):
            return super().load_data(anno_file, obj2vps, debug=debug)
        
        with open(str(anno_file), "r") as f:
            data = []
            for i, line in enumerate(f.readlines()):
                if debug and i==20:
                    break
                data.append(json.loads(line.strip()))

        new_data = []
        sample_idx = 0

        data = tqdm(data, desc="Loading data")
        for i, item in enumerate(data):
            new_item = dict(item)
            new_item["raw_idx"] = i
            new_item["sample_idx"] = sample_idx
            new_item['instr_id'] = f'reverie_scalevln_{item["instr_id"]}'
            new_item['data_type'] = 'reverie_scalevln'
            new_item['objId'] = None
            new_item["path_id"] = None
            new_item["heading"] = item.get("heading", 0)
            new_item['end_vps'] = item['pos_vps']
            del new_item['pos_vps']
            new_data.append(new_item)
            sample_idx += 1

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path'], x['objId']) \
            for x in new_data if 'objId' in x and x['objId'] is not None
        }

        return new_data, gt_trajs