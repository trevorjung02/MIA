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
    # CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
    CACHE_DIR = "/gscratch/zlab/swj0419/huggingface_models"
    
    if "pythia" in name:
        model = GPTNeoXForCausalLM.from_pretrained(name, return_dict=True, 
        cache_dir=CACHE_DIR).cuda()
    elif "gpt2" in name:
            model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, cache_dir=CACHE_DIR).cuda()
            model.config.pad_token_id = model.config.eos_token_id
            model.generation_config.pad_token_id = model.config.eos_token_id
    elif "llama" in name:
        CACHE_DIR = "/gscratch/zlab/swj0419/huggingface_models"
        model = AutoModelForCausalLM.from_pretrained(
           pretrained_model_name_or_path=name,
           torch_dtype=torch.bfloat16,
           trust_remote_code=True,
           low_cpu_mem_usage=True,
           device_map="auto",
           load_in_8bit=True,
            cache_dir=CACHE_DIR
        )
    
    if path:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    if name == "swj0419/7b_finetuned_llama2_3epoch":
        name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def jsonl_to_list(test_data):
    test_json = json.dumps(test_data)
    test_df = pd.read_json(io.StringIO(test_json))['input']
    test_list = test_df.astype(str).tolist()
    return test_list