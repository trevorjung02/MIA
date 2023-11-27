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
    
def load_model(target_name, ref_name, target_path=None, ref_path=None):
    CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
    
    target_model = AutoModelForCausalLM.from_pretrained(target_name, return_dict=True, cache_dir=CACHE_DIR).cuda()
    if target_path:
        ckpt = torch.load(target_path)
        target_model.load_state_dict(ckpt['model_state_dict'])
    target_model.config.pad_token_id = target_model.config.eos_token_id
    target_model.generation_config.pad_token_id = target_model.config.eos_token_id
    
    if "pythia" in ref_name:
        ref_model = GPTNeoXForCausalLM.from_pretrained(ref_name, return_dict=True, cache_dir=CACHE_DIR).cuda()
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_name, return_dict=True, cache_dir=CACHE_DIR).cuda()
        ref_model.config.pad_token_id = ref_model.config.eos_token_id
        ref_model.generation_config.pad_token_id = ref_model.config.eos_token_id
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

def jsonl_to_list(test_data):
    test_json = json.dumps(test_data)
    test_df = pd.read_json(io.StringIO(test_json))['input']
    test_list = test_df.astype(str).tolist()
    return test_list