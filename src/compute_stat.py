from pathlib import Path
from scipy.stats import ttest_ind_from_stats, wasserstein_distance, ks_2samp
import numpy as np
import json
from tqdm import tqdm
import sys


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


if __name__ == "__main__":
    target = sys.argv[1]
    if target == "entropy":
        f = np.mean
    elif target == "logits":
        f = lambda x: np.mean(np.log(x))
    data_path = Path(__file__).parents[1] / target.split('_')[0]
    stat_path = Path(__file__).parents[1] / 'stats'
    models = ['gpt2.json', 'gpt2-medium.json', 'gpt2-large.json', 'gpt2-xl.json',
              'llama2-7b.json', 'llama2-13b.json', 'llama2-70b.json',
              'bert-base-uncased.json', 'bert-large-uncased.json', 'bert-base-cased.json', 'bert-large-cased.json',
              'roberta-base.json', 'roberta-large.json',
              'albert-base-v2.json', 'albert-large-v2.json', 'albert-xlarge-v2.json', 'albert-xxlarge-v2.json',
              'bart-base.json', 'bart-large.json',
              't5-small.json', 't5-base.json', 't5-large.json', 't5-3b.json', 't5-11b.json']
    ret = dict()
    for data_name in ['tc', 'wow', 'cmu', 'faithdial', 'xsum', 'wikibio']:
        ret[data_name] = {'T-test P-value': dict(), 'KS-test P-value': dict(), 'Wasserstein distance': dict()}
        for model in tqdm(models):
            model_name = model.rstrip('.json')
            logits = load_json(data_path / model)
            hallu = list()
            entail = list()
            for dataset, logit in logits.items():
                if data_name in dataset:
                    hallu.extend([f(a) for a in logit['Hallucination'] if len(a) > 0 and not np.isnan(a[0])])
                    entail.extend([f(a) for a in logit['Entailment'] if len(a) > 0 and not np.isnan(a[0])])
            hallu_mean = np.mean(hallu)
            hallu_std = np.std(hallu)
            hallu_obs = len(hallu)
            entail_mean = np.mean(entail)
            entail_std = np.std(entail)
            entail_obs = len(entail)
            stat, p = ttest_ind_from_stats(mean1=hallu_mean, std1=hallu_std, nobs1=hallu_obs,
                                           mean2=entail_mean, std2=entail_std, nobs2=entail_obs)
            ksstat, ksp = ks_2samp(hallu, entail)
            wdist = wasserstein_distance(hallu, entail)
            ret[data_name]['T-test P-value'][model_name] = (stat, p)
            ret[data_name]['KS-test P-value'][model_name] = (ksstat, ksp)
            ret[data_name]['Wasserstein distance'][model_name] = wdist
    write_json(ret, stat_path / f'{target}.json')
