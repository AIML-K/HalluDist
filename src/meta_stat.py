from pathlib import Path
from scipy.stats import ttest_ind_from_stats
import numpy as np
import json
import os
import sys

MODELS = {"Encoder": ["bert", "roberta", "albert"],
          "Decoder": ["gpt2", "llama2"],
          "Enc \\& Dec": ["bart", "t5"]}


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


if __name__ == "__main__":
    target = sys.argv[1]
    print(target)
    p = 0.01
    statistics = load_json(Path(__file__).parents[1] / 'stats' / f'{target}.json')
    ret = dict()
    for model_shape, models in MODELS.items():
        num_models = len([i for i in statistics['tc']['KS-test P-value'].keys() if i.split('-')[0] in models])
        num_data = len(statistics.keys())
        ret[model_shape] = {"sig": {"total": num_models * num_data, "sig": 0}, "ks": []}
        for data, stat in statistics.items():
            for model, t in stat['KS-test P-value'].items():
                if model.split('-')[0] in models and t[1] <= p:
                    ret[model_shape]["sig"]["sig"] += 1
                if model.split('-')[0] in models:
                    ret[model_shape]["ks"].append(t[0])
        ret[model_shape]["sig"]["ratio"] = round(ret[model_shape]["sig"]["sig"] / ret[model_shape]["sig"]["total"], 4)
        ret[model_shape]["ks"] = (round(np.mean(ret[model_shape]["ks"]), 4), round(np.std(ret[model_shape]["ks"]), 4))
    print(ret)
