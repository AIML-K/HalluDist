import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np


MODELS = {"gpt2": ["", "medium", "large", "xl"],
          "llama2": ["7b", "13b", "70b"],
          "roberta": ["base", "large"],
          "albert": ["base", "large", "xlarge", "xxlarge"],
          "bart": ["base", "large"],
          "t5": ["small", "base", "large", "3b", "11b"]}


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


def plot_distribution_modeltype():
    measure = "entropy"
    data_path = Path(__file__).parents[1] / measure
    models = ["gpt2", "llama2-7b", "bert-base-uncased", "roberta-base", "albert-base-v2", "bart-base", "t5-small"]
    data_names = ['tc', 'wow', 'cmu', 'faithdial', 'xsum', 'wikibio']
    for n, model in enumerate(models):
        data = load_json(data_path / f"{model}.json")
        for n2, data_name in enumerate(data_names):
            plt.subplot(len(data_names), len(models), n2 * len(models) + n + 1)
            hallu = list()
            entail = list()
            for k in data.keys():
                if data_name in k:
                    if measure == "logits":
                        hallu.extend([np.mean(np.log(i)) for i in data[k]['Hallucination'] if len(i) > 0])
                        entail.extend([np.mean(np.log(i)) for i in data[k]['Entailment'] if len(i) > 0])
                    else:
                        hallu.extend([np.mean(i) for i in data[k]['Hallucination'] if len(i) > 0])
                        entail.extend([np.mean(i) for i in data[k]['Entailment'] if len(i) > 0])
            xmax = max(hallu + entail)
            xmin = min(hallu + entail)
            plt.hist(hallu, weights=np.ones_like(hallu) / len(hallu), alpha=0.3, bins=30, range=(xmin, xmax), color="r", label='hallucination')
            plt.hist(entail, weights=np.ones_like(entail) / len(entail), alpha=0.3, bins=30, range=(xmin, xmax), color="b", label='entailment')
            plt.axvline(np.mean(hallu), color="r", linestyle="--", linewidth=3)
            plt.axvline(np.mean(entail), color="b", linestyle="--", linewidth=3)
            plt.tick_params(axis='y', left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            if n+1 == len(models) and n2 == 0:
                plt.legend(loc='upper left', bbox_to_anchor=(0.4, 1.5))
            if n == 0:
                plt.ylabel(data_name.upper(), size='large')
            if n2+1 == len(data_names):
                plt.xlabel(model.split('-')[0].upper(), size='large')
    plt.gcf().set_size_inches(25, 10)
    plt.savefig(Path(__file__).parents[1] / 'plots' / "fig1.png")


def plot_size_effect():
    measure = ["entropy", "logits"]
    target_models = ["gpt2", "llama2", "albert", "t5"]
    fig, ax = plt.subplots(1,2)
    for n, m in enumerate(measure):
        data = load_json(Path(__file__).parents[1] / 'stats' / f'{m}.json')
        ret = dict()
        for model_group in target_models:
            stat = list()
            model_size = MODELS[model_group]
            for size in model_size:
                size_stat = list()
                if model_group == "gpt2" and size == "":
                    model = model_group
                else:
                    model = f"{model_group}-{size}"
                    if model_group == "albert":
                        model += "-v2"
                for data_name in data.keys():
                    size_stat.append(data[data_name]["Wasserstein distance"][model])
                stat.append(np.mean(size_stat))
            ret[model_group] = stat
        for k, v in ret.items():
            x = np.linspace(0, 1, len(v))
            v = np.array(v) / v[0]
            ax[n].plot(x, v, label=k)
        x = np.arange(0, 1, 0.499999)
        ax[n].set_xticks(x)
        ax[n].set_xticklabels(['Small', '⟶', 'Large'] , fontsize=14)
        ax[0].set_ylabel("Wasserstein Distance", size=15)
        if m == "entropy":
            ax[n].set_title("Entropy", size=15)
        elif m == "logits":
            ax[n].set_title("Log Token Probability", size=15)
        ax[n].grid()
    plt.gcf().set_size_inches(10, 3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(Path(__file__).parents[1] / 'plots' / "fig2.png")


if __name__ == "__main__":
    plot_distribution_modeltype()
    plot_size_effect()