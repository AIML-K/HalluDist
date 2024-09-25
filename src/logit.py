import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy

from copy import copy
from tqdm import tqdm
from pathlib import Path
import math
import json
import sys
import os

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
print(model_name)

if "gpt2" in model_name:
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda').eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
elif "llama" in model_name:
    from transformers import LlamaTokenizer, LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name, token=os.environ['HF_TOKEN']).to('cuda').eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
elif "bert" in model_name:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
elif "bart" in model_name:
    from transformers import AutoTokenizer, BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained(model_name).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
    pad_id = tokenizer.pad_token_id
elif "t5" in model_name:
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id

HALLU_LABEL = {0: "Entailment", 1: "Hallucination", 2: "Misc"}


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


class HallucinationDataset(Dataset):
    def __init__(self, dataset, data_name):
        self.reference, self.message, self.label = list(), list(), list()
        if 'knowledge' in dataset[data_name]['0'].keys():
            for item in dataset[data_name].values():
                if isinstance(item['history'][0], float) and math.isnan(item['history'][0]):
                    history = ""
                else:
                    history = '\n'.join(item['history'])
                reference = f"""Knowledge: {item['knowledge']}
                History: {history}
                Message: """
                self.reference.append(reference)
                self.message.append(item["message"])
                self.label.append(item["label"])
        else:
            for item in dataset[data_name].values():
                reference = f"""Reference: {item['reference']}
                Message: """
                self.reference.append(reference)
                self.message.append(item["message"])
                self.label.append(item["label"])


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        reference = self.reference[idx]
        message = self.message[idx]
        label = self.label[idx]
        return reference, message, label


def end_compute_probs(ref, mes):
    with torch.no_grad():
        enc_tokens = tokenizer(ref, return_tensors='pt', truncation=True, padding=True).to('cuda')
        if 't5' in model_name:
            dec_tokens = tokenizer(['<pad> ' + i for i in mes], return_tensors='pt', truncation=True, padding=True).to('cuda')
        elif 'bart' in model_name:
            dec_tokens = tokenizer(['</s> ' + i for i in mes], return_tensors='pt', truncation=True, padding=True).to(
                'cuda')
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


def auto_compute_probs(ref, mes):
    with torch.no_grad():
        texts = [str(i) + ' ' + str(j) for i, j in zip(ref, mes)]
        tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to('cuda')
        logits = model(**tokens).logits
        probs = logits.softmax(dim=2)
    ref_tokens = tokenizer(ref).input_ids
    logits = list()
    entropies = list()
    for prob, ref_token, ids in zip(probs, ref_tokens, tokens.input_ids):
        ids = ids[len(ref_token):]
        ids = ids[ids != pad_id]
        prob = prob[len(ref_token)-1:]
        ret1 = list()
        ret2 = list()
        for n, i in enumerate(ids):
            ret1.append(prob[n, i].item())
            ret2.append(float(entropy(prob[n].cpu())))
        logits.append(ret1)
        entropies.append(ret2)
    return logits, entropies


def mlm_compute_probs(ref, mes):
    ref = ref[0]
    mes = mes[0]
    text = ref + " " + mes
    input_tokens = tokenizer(text, truncation=True).input_ids
    if len(input_tokens) == 512: # Too long => Pass
        return [], []
    mes_begin = len(tokenizer(ref).input_ids) - 1
    tokens = list()
    for n, i in enumerate(input_tokens[mes_begin:-1]):
        tmp = copy(input_tokens)
        tmp[mes_begin+n] = mask_id
        tokens.append(tmp)
    with torch.no_grad():
        tokens = torch.tensor(tokens).to("cuda")
        if tokens.shape[0] > 80: # Too long => divide batch to avoid OOM
            probs = list()
            for token in [tokens[:tokens.shape[0] // 3,:], tokens[tokens.shape[0] // 3:(tokens.shape[0] // 3) * 2,:],
                          tokens[(tokens.shape[0] // 3) * 2:,:]]:
                logit = model(input_ids=token).logits
                prob = torch.softmax(logit, dim=2).to('cpu')
                probs.append(prob)
            probs = torch.concat(probs)
        else:
            logits = model(input_ids=tokens).logits
            probs = torch.softmax(logits, dim=2)
    logits = list()
    entropies = list()
    for n, i in enumerate(input_tokens[mes_begin:-1]):
        p = probs[n][mes_begin+n][i].item()
        e = entropy(probs[n][mes_begin+n].cpu())
        logits.append(p)
        entropies.append(float(e))
    return logits, entropies


if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / 'hallu_data'
    logits_path = Path(__file__).parents[1] / 'logits'
    entropies_path = Path(__file__).parents[1] / 'entropy'
    logits = dict()
    entropies = dict()
    data = load_json(data_path / 'hallu_dataset.json')
    for n, data_name in enumerate(data.keys()):
        print(f"{n+1} / {len(data.keys())} Processing")
        logits[data_name] = {"Hallucination": list(), "Entailment": list(), "Misc": list()}
        entropies[data_name] = {"Hallucination": list(), "Entailment": list(), "Misc": list()}
        dataloader = DataLoader(HallucinationDataset(data, data_name), batch_size=batch_size, shuffle=False)
        for ref, mes, label in tqdm(dataloader):
            if 'bert' in model_name:
                probs, entrp = mlm_compute_probs(ref, mes)
                logits[data_name][HALLU_LABEL[label.item()]].append(probs)
                entropies[data_name][HALLU_LABEL[label.item()]].append(entrp)
            elif 'bart' in model_name or 't5' in model_name:
                probs, entrp = end_compute_probs(ref, mes)
                for i, j, k in zip(probs, entrp, label.tolist()):
                    logits[data_name][HALLU_LABEL[k]].append(i)
                    entropies[data_name][HALLU_LABEL[k]].append(j)
            else:
                probs, entrp = auto_compute_probs(ref, mes)
                for i, j, k in zip(probs, entrp, label.tolist()):
                    logits[data_name][HALLU_LABEL[k]].append(i)
                    entropies[data_name][HALLU_LABEL[k]].append(j)
        if "/" in model_name:
            write_json(logits, logits_path / f"{model_name.split('/')[1]}.json")
            write_json(entropies, entropies_path / f"{model_name.split('/')[1]}.json")
        else:
            write_json(logits, logits_path / f"{model_name}.json")
            write_json(entropies, entropies_path / f"{model_name}.json")
