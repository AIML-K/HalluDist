from transformers import GPT2TokenizerFast, GPT2LMHeadModel, T5TokenizerFast, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import math
import os
import sys


dataset = sys.argv[1]


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


class HallucinationDataset(Dataset):
    def __init__(self, dataset, data_split):
        self.index, self.reference, self.message = list(), list(), list()
        if 'knowledge' in dataset[data_split]['0'].keys():
            for key, item in dataset[data_split].items():
                if len(item['history']) == 0 or (isinstance(item['history'][0], float) and math.isnan(item['history'][0])):
                    history = ""
                else:
                    history = ' / '.join(item['history'])
                reference = f"""Knowledge: {item['knowledge']} / History: {history} / Message: """
                self.index.append(key)
                self.reference.append(reference)
                self.message.append(item["message"])
        else:
            for key, item in dataset[data_split].items():
                reference = f"""Reference: {item['reference']} / Message: """
                self.index.append(key)
                self.reference.append(reference)
                self.message.append(item["message"])

    def __len__(self):
        return len(self.message)

    def __getitem__(self, idx):
        index = self.index[idx]
        reference = self.reference[idx]
        message = self.message[idx]
        return index, reference, message


if __name__ == "__main__":
    data_path = Path(__file__).parents[2] / 'chat_data'
    tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=1024)
    batch_size = 16
    data = load_json(data_path / f'{dataset}.json')
    dataloader = list(DataLoader(HallucinationDataset(data, 'test'), batch_size=batch_size, shuffle=False))
    for target in os.listdir(Path(__file__).parents[2] / 'train' / dataset):
        if target.endswith('csv') or target == 'ctrl':
            continue
        for trial in range(5):
            print(f'==== Generating {dataset} - {target} - Trial {trial} ====')
            model_path = Path(__file__).parents[2] / 'train' / dataset / target / f'{trial}trial'
            model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')
            ret = dict()
            for idx, ref, _ in tqdm(dataloader):
                ipt = tokenizer(ref, return_tensors='pt', truncation=True, padding=True).to('cuda')
                opt = model.generate(**ipt, max_length=100)
                opt = tokenizer.batch_decode(opt, skip_special_tokens=True)
                for i, j in zip(idx, opt):
                    ret[i] = j.strip()
            write_json(ret, model_path / 'generate.json')
