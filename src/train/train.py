from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import math
import json
import sys

loss_fn = CrossEntropyLoss(reduction='none')
model_name = 't5-small'
batch_size = 4

cummulation_step = 16 / batch_size
dataset = sys.argv[1]
target = sys.argv[2]


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


class HallucinationDataset(Dataset):
    def __init__(self, dataset, data_split, weighting=True):
        self.reference, self.weight, self.message = list(), list(), list()
        if weighting:
            metrics = torch.tensor([i[target] for i in dataset[data_split].values()])
            if target == 'entropy':
                metrics = - metrics
            weight = torch.softmax(metrics, dim=0) * len(metrics)
        if 'knowledge' in dataset[data_split]['0'].keys():
            for n, item in enumerate(dataset[data_split].values()):
                if len(item['history']) == 0 or (isinstance(item['history'][0], float) and math.isnan(item['history'][0])):
                    history = ""
                else:
                    history = ' / '.join(item['history'])
                reference = f"""Knowledge: {item['knowledge']} / History: {history} / Message: """
                if data_split == 'train':
                    self.reference.append(reference)
                    if weighting:
                        self.weight.append(weight[n])
                    else:
                        self.weight.append(1)
                    self.message.append(item["message"])
                else:
                    self.reference.append(reference)
                    self.weight.append(1)
                    self.message.append(item["message"])
        else:
            for n, item in enumerate(dataset[data_split].values()):
                reference = f"""Reference: {item['reference']} / Message: """
                if data_split == 'train':
                    self.reference.append(reference)
                    if weighting:
                        self.weight.append(weight[n])
                    else:
                        self.weight.append(1)
                    self.message.append(item["message"])
                else:
                    self.reference.append(reference)
                    self.weight.append(1)
                    self.message.append(item["message"])

    def __len__(self):
        return len(self.message)

    def __getitem__(self, idx):
        reference = self.reference[idx]
        weight = self.weight[idx]
        message = self.message[idx]
        return reference, weight, message


def train(model, optimizer, tokenizer, dataloader):
    print_loss = 0
    cummulation = 0
    pbar = tqdm(dataloader)
    for n, (ref, wei, mes) in enumerate(pbar):
        input_tokens = tokenizer(ref, return_tensors='pt', truncation=True, padding=True).to('cuda')
        label_tokens = tokenizer(mes, return_tensors='pt', truncation=True, padding=True).input_ids.to('cuda')
        label_tokens[label_tokens == tokenizer.pad_token_id] = -100
        logits = model(**input_tokens, labels=label_tokens).logits
        loss = loss_fn(logits.transpose(1,2), label_tokens)
        loss *= wei.unsqueeze(1).to('cuda')
        loss = loss.sum() / (label_tokens != -100).sum() / cummulation_step
        if not np.isnan(loss.item()):
            loss.backward()
            print_loss += loss.item() * cummulation_step
            cummulation += 1
        if cummulation % cummulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        pbar.set_description(f"Train Loss: {round(print_loss / (n+1), 5)}")


def valid(model, tokenizer, dataloader):
    print_loss = 0
    pbar = tqdm(dataloader)
    for n, (ref, wei, mes) in enumerate(pbar):
        input_tokens = tokenizer(ref, return_tensors='pt', truncation=True, padding=True).to('cuda')
        label_tokens = tokenizer(mes, return_tensors='pt', truncation=True, padding=True).input_ids.to('cuda')
        label_tokens[label_tokens == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            loss = model(**input_tokens, labels=label_tokens).loss.item() / cummulation_step
        if np.isnan(loss):
            pass
        else:
            print_loss += loss
        pbar.set_description(f"Validation Loss: {round(print_loss / (n+1), 5)}")


if __name__ == "__main__":
    for trial in range(5):
        model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda')
        tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=1024)
        optimizer = AdamW(model.parameters(), 1e-4)
        data_path = Path(__file__).parents[2] / 'chat_data'
        print(f"=========== Training {dataset} - {target} - Trial {trial} ==============")
        model_path = Path(__file__).parents[2] / 'train' / dataset / target / f'{trial}trial'
        data = load_json(data_path / f'{dataset}_metric.json')
        if target == "none":
            w = False
        else:
            w = True
        train_dataloader = list(DataLoader(HallucinationDataset(data, "train", weighting=w), batch_size=batch_size, shuffle=True))
        valid_dataloader = list(DataLoader(HallucinationDataset(data, "valid", weighting=w), batch_size=batch_size, shuffle=False))
        for epoch in range(3):
            model.train()
            train(model, optimizer, tokenizer, train_dataloader)
            model.eval()
            valid(model, tokenizer, valid_dataloader)
        model.save_pretrained(model_path)
