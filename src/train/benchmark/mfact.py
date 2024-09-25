from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch

from tqdm import tqdm
from pathlib import Path
import json
import sys


target_data = sys.argv[1]
batch_size = 32
tokenizer = AutoTokenizer.from_pretrained('yfqiu-nlp/mFACT-en_XX')
model = AutoModelForSequenceClassification.from_pretrained('yfqiu-nlp/mFACT-en_XX').to('cuda')


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


if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / 'chat_data'
    logits = list()
    data = load_json(data_path / f'{target_data}.json')
    for data_split in data.keys():
        dataloader = DataLoader(HallucinationDataset(data, data_split), batch_size=batch_size, shuffle=False)
        for ref, mes in tqdm(dataloader):
            tokens = tokenizer(ref, mes, padding=True, truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                probs = model(**tokens).logits.softmax(dim=1)[:,1]
            logits.extend([i.item() for i in probs])
        for k, i in zip(data[data_split].keys(), logits):
            data[data_split][k]['mfact'] = i
    write_json(data, data_path / f'{target_data}_mfact.json')