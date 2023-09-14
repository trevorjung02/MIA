import logging
logging.basicConfig(level='ERROR')
import statistics
import random
import argparse
import numpy as np
from pathlib import Path
from pprint import pprint
from sentence_transformers import SentenceTransformer, util
import sys
import torch
import zlib
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
from ipdb import set_trace as bp
import json
from collections import defaultdict

import os
import scipy.stats

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools

import pickle
from plots import fig_fpr_tpr
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl
from detectGPT import DetectGPTPerturbation

# load detectGPT
detect_gpt = DetectGPTPerturbation()

# Returns the perplexity of the batch (in this case, just one sentence), and the list of the probabilities the model gives to each labeled token
def calculatePerplexity(sentence, model, tokenizer, gpu):

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob


def inference(model1, model2, tokenizer1, tokenizer2, text, ex):
    # ex["pred"] = {}
    pred = {}

    # perplexity of large and small models
    p_target, all_prob = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_ref, _ = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
    p_lower, _= calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    # detectGPT 
    perturbed_text = detect_gpt.perturb(text)
    p_perturb, _ = calculatePerplexity(perturbed_text, model1, tokenizer1, gpu=model1.device)

    # detectGPT on reference
    p_ref_perturb, _ = calculatePerplexity(perturbed_text, model2, tokenizer2, gpu=model2.device)

    # ppl
    pred["ppl"] = p_target
    # Ratio of log ppl of large and small models
    pred["log_ppl/log_ref_ppl"] = (np.log(p_target)/np.log(p_ref)).item()
    
    pred[f"log_ppl/log_ppl_perturb"] = (np.log(p_target)/np.log(p_perturb)).item()

    pred["RMIA"] = (np.log(p_target) / np.log(p_perturb) * np.log(p_ref_perturb) / np.log(p_ref)).item()

    # Ratio of log ppl of lower-case and normal-case
    pred["log_ppl/log_lower_ppl"] = (np.log(p_target) / np.log(p_lower)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["log_ppl/zlib"] = np.log(p_target)/zlib_entropy
    # token mean and var
    pred["mean"] = -np.mean(all_prob).item()
    pred["var"] = np.var(all_prob).item()
    ex["pred"] = pred

    return ex

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, output_dir, col_name):
    all_output = []
    for ex in tqdm(test_data): # [:100]
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex)
        all_output.append(new_ex)
    return all_output

if __name__ == '__main__':
    sourcemodel2target = {"gpt2-large": ["gpt2"]}
    source_name = "wiki" # llama
    for k, v in tqdm(sourcemodel2target.items()):
        name_part1 = model_name_1 = k
        # create output directory name:
        if "/" in k:
            name_part1 = k.split("/")[-1]
        for model_name_2 in v:
            name_part2 = model_name_2
            if "/" in model_name_2:
                name_part2 = name_part2.split("/")[-1]
            model1, model2, tokenizer1, tokenizer2 = load_model(model_name_1, model_name_2)
# =============================================================================
            for col_name in ["input"]: # , "paraphrase", "drop_0.1",  "drop_0.3",  "drop_0.5" 
                for data_name in ["final_128"]: # "final_64", "final_32", 
                    # output path
                    output_dir = f"output/{source_name}/{data_name}/{name_part1}/{name_part2}/{col_name}"
                    print("output_dir: ", output_dir)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    # input path
                    # input_path = f"/fsx-instruct-opt/swj0419/attack/wikidata/{source_name}_data/{data_name}.jsonl"
                    input_path = f"data/{data_name}.jsonl"
                    data = load_jsonl(input_path)

                    all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, output_dir, col_name)
                    '''
                    dump and read data 
                    '''
                    dump_jsonl(all_output, f"{output_dir}/output.jsonl")
                    all_output = read_jsonl(f"{output_dir}/output.jsonl")
                    # bp()
                    '''
                    plot
                    '''
                    fig_fpr_tpr(all_output, output_dir)
                    # compute_topkmean(all_output, output_dir)
                    print("===============================")

        
    