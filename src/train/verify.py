from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from summac.model_summac import SummaCZS
from rouge import Rouge
from bert_score import BERTScorer
from string2string.similarity.bartscore import BARTScore
import torch

from pathlib import Path
from tqdm import tqdm
import json
import os
import sys


dataset = sys.argv[1]


def check_empty(string):
    if string.strip().strip('.,!?') == "":
        return "no response"
    else:
        return string


class HallucinationDataset(Dataset):
    def __init__(self, dataset, generation):
        self.index, self.knowledge, self.reference, self.generation = list(), list(), list(), list()
        for (key, item1), item2 in zip(dataset.items(), generation.values()):
            self.index.append(key)
            if 'knowledge' in dataset['0'].keys():
                self.knowledge.append(check_empty(item1['knowledge']))
            else:
                self.knowledge.append(check_empty(item1['reference']))
            self.reference.append(check_empty(item1['message']))
            self.generation.append(check_empty(item2))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        index = self.index[idx]
        knowledge = self.knowledge[idx]
        reference = self.reference[idx]
        generation = self.generation[idx]
        return index, knowledge, reference, generation


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


if __name__ == "__main__":
    data_path = Path(__file__).parents[2] / 'chat_data'
    rouge_model = Rouge()
    bert_model = BERTScorer(lang="en", device="cuda", rescale_with_baseline=True)
    bart_model = BARTScore(device="cuda")
    summac_model = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
    factkb_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    factkb_model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB").to('cuda').eval()
    data = load_json(data_path / f'{dataset}.json')['test']
    for target in os.listdir(Path(__file__).parents[2] / 'train' / dataset):
        if target.endswith('csv'):
            continue
        for trial in range(5):
            print(f'==== Verifying {dataset} - {target} - Trial {trial} ====')
            model_path = Path(__file__).parents[2] / 'train' / dataset / target / f'{trial}trial'
            generated = load_json(model_path / 'generate.json')
            result = dict()
            loader = list(DataLoader(HallucinationDataset(data, generated), shuffle=False, batch_size=16))
            for idx, kdg, ref, gen in tqdm(loader):
                with torch.no_grad():
                    summac = summac_model.score(kdg, gen)['scores']
                    tokens = factkb_tokenizer(gen, kdg, truncation=True, padding=True, return_tensors='pt').to('cuda')
                    factkb = factkb_model(**tokens).logits.softmax(dim=1)[:, 1].tolist()
                    rouge = [i['rouge-l']['f'] for i in rouge_model.get_scores(ref, gen)]
                    bert = bert_model.score(ref, gen)[2].tolist()
                    bart = bart_model.compute(ref, gen)['score'].tolist()
                    for i, su, kb, ro, be, ba in zip(idx, summac, factkb, rouge, bert, bart):
                        result[i] = {'summac': su, 'factkb': kb, 'rouge': ro, 'bertscore': be, 'bartscore': ba}
            write_json(result, model_path / 'metric.json')
