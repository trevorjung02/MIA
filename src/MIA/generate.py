import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import torch
from argparse import ArgumentParser
import os
import pandas as pd
import io
import numpy as np
import gc

from load_utils import create_new_dir, dump_jsonl

def load_model(name, path=None):
    CACHE_DIR = "/gscratch/zlab/swj0419/huggingface_models"

    if "pythia" in name:
        model = GPTNeoXForCausalLM.from_pretrained(name, return_dict=True, 
        cache_dir=CACHE_DIR).cuda()
    elif "gpt2" in name:
        model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, cache_dir=CACHE_DIR).cuda()
    elif "llama" in name:
        model = AutoModelForCausalLM.from_pretrained(
           pretrained_model_name_or_path=name,
           torch_dtype=torch.bfloat16,
           trust_remote_code=True,
           low_cpu_mem_usage=True,
           device_map="auto",
           load_in_8bit=True,
            cache_dir=CACHE_DIR
        )
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    
    if path:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    if "swj0419" in name:
        tokenizer = None
    elif "gpt2" in name:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines()

    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]

    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs) - 1):
        start = start_idxs[i]
        end = start_idxs[i + 1]
        if "WARC-Identified-Content-Language: eng" in lines[start + 7]:
            count_eng += 1
            for j in range(start + 10, end):
                all_eng += lines[j]

    return all_eng
    
def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--target_model', type=str, default='gpt2', help="Model name of the target model, gpt2 by default")
    parser.add_argument('--n_per_prompt', type=int, default=1, help="Number of candidates to generate per prompt")
    parser.add_argument('--n_candidates', type=int, default=100000, help="Number of candidates to generate")
    parser.add_argument('--attack_args_path', type=str, help="Path to json file containing generation args")
    args = parser.parse_args()
    return vars(args)
    
def main():
    args = get_cli_args()
    model, tokenizer = load_model(args['target_model'])
    target_dir = create_new_dir(f"kNNLM_privacy/generated/{args['target_model']}")
    target_file = os.path.join(target_dir, "data.jsonl")
    args_file = os.path.join(target_dir, "args.json")

    cc = parse_commoncrawl("kNNLM_privacy/prompts/crawl/commoncrawl.warc.wet")
    prompt_list = []
    while len(prompt_list) < args['n_candidates'] // args['n_per_prompt'] :
        r = np.random.randint(0, len(cc))
        prompt = " ".join(cc[r : r + 50].split(" ")[1:-1])
        if len(prompt) > 0:
            prompt_list.append(prompt)

    with open(args['attack_args_path']) as f:
        attack_args = json.load(f)
    args['attack_args'] = attack_args
    with open(args_file, 'w') as f:
        f.write(json.dumps(args))
        
    output_dicts = []
    for ip, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        print(ip, prompt)
        if ip % 10 == 0:
            gc.collect()
        for _ in range(args['n_per_prompt'] // attack_args['num_return_sequences']):
            generated = model.generate(
                tokenizer.encode(prompt, return_tensors="pt").to(model.device),
                return_dict_in_generate=True,
                **attack_args
            )
    
            generated_ids = generated["sequences"]
            sentences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
            for i, beam_output in enumerate(generated_ids):
                row = {
                    'ip': ip,
                    'i': i,
                    'input': sentences[i].strip()
                }
                output_dicts.append(row)
    
    dump_jsonl(output_dicts, target_file)

if __name__ == "__main__":
    main()