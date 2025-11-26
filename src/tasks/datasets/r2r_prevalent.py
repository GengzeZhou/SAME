import json
import numpy as np
from tqdm import tqdm
from .r2r import R2RDataset

class R2RAugDataset(R2RDataset):
    name = "r2r_prevalent"

    def load_data(self, anno_file, max_instr_len=200, debug=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """
        if str(anno_file).endswith(".json"):
            # Load from .json file
            with open(str(anno_file), "r") as f:
                data = json.load(f)
            new_data = []

            data = tqdm(data, desc="Loading data")
            for i, item in enumerate(data):
                new_item = dict(item)
                new_item['raw_idx'] = i
                new_item['sample_idx'] = i
                new_item['instr_id'] = f'r2r_prevalent_{item["path_id"]}'

                new_item['instruction'] = item['instructions'][0]
                del new_item['instructions']

                new_item['instr_encoding'] = item['instr_enc'][:max_instr_len]
                del new_item['instr_enc']

                new_item['data_type'] = 'r2r_prevalent'
                new_data.append(new_item)
        else:
            # Load from .jsonl file
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
                new_item['data_type'] = 'r2r_prevalent'
                new_item["instr_id"] = f'r2r_prevalent_{item["path_id"]}'
                new_item["heading"] = item.get("heading", 0)
                new_item["instr_encoding"] = item['instr_enc'][:max_instr_len]
                del new_item['instr_enc']
                new_data.append(new_item)
                sample_idx += 1

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }
        return new_data, gt_trajs

