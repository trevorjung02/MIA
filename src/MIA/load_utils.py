import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from argparse import ArgumentParser
import os

def create_new_dir(dir: str) -> str:
    if not os.path.exists(dir):
        num = 1
    else:
        files = set(os.listdir(dir))
        num = len(files)+1
        while f"run_{num}" in files:
            num += 1
    new_dir = os.path.join(dir, f"run_{num}")
    os.makedirs(new_dir, exist_ok=False)
    return new_dir

def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)

    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]
    
def load_model(target_name, ref_name, target_path=None, ref_path=None):
    CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
    
    target_model = AutoModelForCausalLM.from_pretrained(target_name, return_dict=True, cache_dir=CACHE_DIR).cuda()
    if target_path:
        ckpt = torch.load(target_path)
        target_model.load_state_dict(ckpt['model_state_dict'])
    
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name, return_dict=True, cache_dir=CACHE_DIR).cuda()
    if ref_path:
        ckpt = torch.load(ref_path)
        ref_model.load_state_dict(ckpt['model_state_dict'])

    target_model.eval()
    ref_model.eval()
    tokenizer1 = AutoTokenizer.from_pretrained(target_name)
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2 = AutoTokenizer.from_pretrained(target_name)
    tokenizer2.pad_token = tokenizer2.eos_token

    return target_model, ref_model, tokenizer1, tokenizer2