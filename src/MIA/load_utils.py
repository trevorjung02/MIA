import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

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
    
def load_model(name1, name2):
    tokenizer = AutoTokenizer.from_pretrained(name1)
    # tokenizer.padding_side = "left" 
    # tokenizer.pad_token = tokenizer.eos_token , load_in_8bit=True , load_in_8bit=True
    CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
    if "gpt2" in name1:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto', cache_dir=CACHE_DIR)
        model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto', cache_dir=CACHE_DIR)
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto', torch_dtype=torch.float16, cache_dir=CACHE_DIR)
        model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto', torch_dtype=torch.float16, cache_dir=CACHE_DIR)
        # torch_type = torch.float16
    # model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
    # model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto')
    # , load_in_8bit=True dtype=float16
    model1.eval()
    model2.eval()
    tokenizer1 = AutoTokenizer.from_pretrained(name1)
    tokenizer2 = AutoTokenizer.from_pretrained(name1)
    return model1, model2, tokenizer1, tokenizer2