import pandas as pd
from pathlib import Path
import numpy as np
import json
import os
import sys


dataset = sys.argv[1]
metrics = ['summac', 'rouge', 'bertscore', 'bartscore']
# If Q2 metrics are ready, use below
# metrics = ['Q2_no_nli', 'Q2', 'summac', 'faithcritic', 'factkb', 'rouge', 'bertscore', 'bartscore']


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


if __name__ == "__main__":
    ret = [[''] + metrics]
    for target in os.listdir(Path(__file__).parents[2] / 'train' / dataset):
        if target.endswith('csv'):
            continue
        model_path = Path(__file__).parents[2] / 'train' / dataset / target
        q2_path = Path(__file__).parents[2] / 'train' / 'q2' / dataset / target
        metric_dict = {k:[] for k in metrics}
        for trial in range(5):
            computed_metric = load_json(model_path / f'{trial}trial' / 'metric.json')
            if metrics in metrics:
                computed_nli = pd.read_csv(q2_path / f'{trial}trial' / 'nli.csv')
            for i in metrics:
                if 'Q2' in i:
                    score = computed_nli[i].mean()
                    metric_dict[i].append(score)
                else:
                    score = np.mean([item[i] for item in computed_metric.values()])
                    metric_dict[i].append(score)
        line = [target]
        line2 = ['']
        for i in metrics:
            mean = round(np.mean(metric_dict[i]), 4)
            std = round(np.std(metric_dict[i]), 2)
            line.append(f'{mean}' + ' \std{(' + str(std) + ')}')
        ret.append(line)
    ret = pd.DataFrame(ret)
    ret.to_csv(Path(__file__).parents[2] / 'train' / dataset / 'result.csv', header=False, index=False)