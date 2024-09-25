from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import entropy

from copy import copy
from tqdm import tqdm
from pathlib import Path
import math
import json
import sys


target_data = sys.argv[1]
model_name = 't5-small'
batch_size = 32

model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
pad_id = tokenizer(tokenizer.pad_token, add_special_tokens=False).input_ids[0]

HALLU_LABEL = {0: "Entailment", 1: "Hallucination", 2: "Misc"}


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


class HallucinationDataset(Dataset):
    def __init__(self, dataset, data_split):
        self.reference, self.message = list(), list()
        if 'knowledge' in dataset[data_split]['0'].keys():
            for item in dataset[data_split].values():
                if (len(item['history']) == 0) or (isinstance(item['history'][0], float) and math.isnan(item['history'][0])):
                    history = ""
                else:
                    history = '\n'.join(item['history'])
                reference = f"""Knowledge: {item['knowledge']}
                History: {history}
                Message: """
                self.reference.append(reference)
                self.message.append(item["message"])
        else:
            for item in dataset[data_split].values():
                reference = f"""Reference: {item['reference']}
                Message: """
                self.reference.append(reference)
                self.message.append(item["message"])


    def __len__(self):
        return len(self.message)

    def __getitem__(self, idx):
        reference = self.reference[idx]
        message = self.message[idx]
        return reference, message


def end_compute_probs(ref, mes):
    with torch.no_grad():
        enc_tokens = tokenizer(ref, return_tensors='pt', truncation=True, padding=True).to('cuda')
        dec_tokens = tokenizer(['<pad> ' + i for i in mes], return_tensors='pt', truncation=True, padding=True).to('cuda')
        logits = model(**enc_tokens, decoder_input_ids=dec_tokens.input_ids).logits
        probs = logits.softmax(dim=2)
    logits = list()
    entropies = list()
    for prob, ids in zip(probs, dec_tokens.input_ids):
        ids = ids[ids != pad_id]
        ret1 = list()
        ret2 = list()
        for n, i in enumerate(ids):
            ret1.append(prob[n, i].item())
            ret2.append(float(entropy(prob[n].cpu())))
        logits.append(ret1)
        entropies.append(ret2)
    return logits, entropies


if __name__ == "__main__":
    data_path = Path(__file__).parents[2] / 'chat_data'
    logits = list()
    entropies = list()
    data = load_json(data_path / f'{target_data}.json')
    for data_split in data.keys():
        dataloader = DataLoader(HallucinationDataset(data, data_split), batch_size=batch_size, shuffle=False)
        for ref, mes in tqdm(dataloader):
            probs, entrp = end_compute_probs(ref, mes)
            logits.extend([np.mean(np.log(i)) for i in probs])
            entropies.extend([np.mean(i) for i in entrp])
        for k, i, j in zip(data[data_split].keys(), logits, entropies):
            data[data_split][k]['logits'] = i
            data[data_split][k]['entropy'] = j
    write_json(data, data_path / f'{target_data}_metric.json')
