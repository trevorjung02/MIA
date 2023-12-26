import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import torch
from argparse import ArgumentParser
import os
import pandas as pd
import io

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
    
def load_model(name, path=None):
    CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'

    if "pythia" in name:
        model = GPTNeoXForCausalLM.from_pretrained(name, return_dict=True, 
        cache_dir=CACHE_DIR).cuda()
    else:
        if "gpt2" in name:
            model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, cache_dir=CACHE_DIR).cuda()
            model.config.pad_token_id = model.config.eos_token_id
            model.generation_config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, cache_dir=CACHE_DIR, 
            device_map='auto',
            load_in_8bit=True)
    
    if path:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def jsonl_to_list(test_data):
    test_json = json.dumps(test_data)
    test_df = pd.read_json(io.StringIO(test_json))['input']
    test_list = test_df.astype(str).tolist()
    return test_list