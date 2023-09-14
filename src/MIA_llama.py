import logging
logging.basicConfig(level='ERROR')
import statistics
import random
import argparse
import numpy as np
from pathlib import Path
from pprint import pprint
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import sys
import numpy as np
import torch
import zlib
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
from ipdb import set_trace as bp
import json
from collections import defaultdict
import statistics

import os
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools

# Look at me being proactive!
import matplotlib
import pickle
import statistics
from detectGPT import DetectGPTPerturbation
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# load detectGPT
detect_gpt = DetectGPTPerturbation()

# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.001)[0][-1]]
    # bp()
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc,acc, low


def fig_fpr_tpr(all_output, output_dir):
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        # bp()
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4,3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            # print(metric)
            # print(predictions)
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n'%(legend, auc, acc, low))

    # auc to google
    with open(f"{output_dir}/auc_google.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            # print(metric)
            # print(predictions)
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write(f'{legend}	{round(auc, 3)}\n')
# 0.6619	0.6354

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")
    # plt.show()





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

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    print("idxs: ", idxs)
    acc_list = []
    for id in idxs:
        if id < 10:
            acc_list.append(1)
        else:
            acc_list.append(0)
    print("acc: ", statistics.mean(acc_list))


sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def sample_generation(sentence, model, tokenizer, gpu):
    half_sentence_index = int(len(sentence.split())*0.2)
    # half_sentence_index = 8
    prefix = " ".join(sentence.split()[:half_sentence_index])
    continuation = " ".join(sentence.split()[half_sentence_index:])
    input_ids = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    output = model.generate(input_ids, max_new_tokens=len(sentence.split())-half_sentence_index, num_return_sequences=1, num_beams=1, pad_token_id=tokenizer.eos_token_id)
    complete_generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = complete_generated_text[len(prefix):]
    
    #Compute embedding for both lists
    embeddings1 = sim_model.encode([continuation], convert_to_tensor=True)
    embeddings2 = sim_model.encode([generated_text], convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    score = cosine_scores.item()

    # # # compute prefix probability
    # prefix_p, n_tok_prefix, _ = calculatePerplexity(prefix, model, tokenizer, gpu=model.device)

    # # compute contination ppl
    
    # # compute generated probability
    # complete_generated_text_p, n_tok_generated, _ = calculatePerplexity(complete_generated_text, model, tokenizer, gpu=model.device)
    # bp()
    return score, complete_generated_text # complete_generated_text_p, prefix_p

    '''
    emb similarity 
    #Compute embedding for both lists
    embeddings1 = sim_model.encode([continuation], convert_to_tensor=True)
    embeddings2 = sim_model.encode([generated_text], convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    score = cosine_scores.item()
    '''

     

def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)

    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), probabilities.shape[0], all_prob, input_ids_processed



def inference(model1, model2, tokenizer1, tokenizer2, text, ex):
    # ex["pred"] = {}
    pred = {}

    # perplexity of large and small models
    p1, n_tok, all_prob, input_ids_processed = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_ref, _, all_prob_ref, _ = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
    p_lower, _, _, _ = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    # detectGPT 
    perturbed_text = detect_gpt.perturb(text)
    p_perturb, _, _, _ = calculatePerplexity(perturbed_text, model1, tokenizer1, gpu=model1.device)
    pred[f"ppl/ppl_perturb"] = p1/p_perturb

    # detectGPT on reference
    p_ref_perturb, _, _, _ = calculatePerplexity(perturbed_text, model2, tokenizer2, gpu=model2.device)

    # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models
    pred["ppl/s_ppl"] = np.log(p1)/np.log(p_ref).item()

    pred["RMIA"] = np.log(p1) / np.log(p_perturb) * np.log(p_ref_perturb) / np.log(p_ref).item()

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/low_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    # token mean and var
    pred["mean"] = -np.mean(all_prob).item()
    pred["var"] = np.var(all_prob).item()
    ex["pred"] = pred

    return ex


def sig(x, t=1):
    z = 1/(1 + np.exp(-x*t))
    return z


def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, output_dir, col_name):
    all_output = []
    for ex in tqdm(test_data): # [:100]
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex)
        all_output.append(new_ex)
    return all_output



    
def compute_ppl(all_output, output_dir):
    seen_data_avg = []
    unseen_data_avg = []
    for ex in all_output:
        if ex["label"] == 0:
            unseen_data_avg.append(ex["pred"]["ppl"])
        elif ex["label"] == 1:
            seen_data_avg.append(ex["pred"]["ppl"])
    print("seen_data_avg: ", statistics.mean(seen_data_avg))
    print("unseen_data_avg: ", statistics.mean(unseen_data_avg))
    with open(f"{output_dir}/ppl.txt","w") as f:
        f.write(f"seen_data_avg: {statistics.mean(seen_data_avg)}\n")
        f.write(f"unseen_data_avg: {statistics.mean(unseen_data_avg)}\n")

    

def compute_topkmean(all_output, output_dir):
    seen_data_avg = []
    unseen_data_avg = []
    for ex in all_output:
        if ex["label"] == 0:
            unseen_data_avg.append(ex["pred"]["topk_mean"])
        elif ex["label"] == 1:
            seen_data_avg.append(ex["pred"]["topk_mean"])
    print("topkmean:seen_data_avg: ", statistics.mean(seen_data_avg))
    print("topkmean:unseen_data_avg: ", statistics.mean(unseen_data_avg))
    with open(f"{output_dir}/var.txt","w") as f:
        f.write(f"topkmean:seen_data_avg: {statistics.mean(seen_data_avg)}\n")
        f.write(f"topkmean:unseen_data_avg: {statistics.mean(unseen_data_avg)}\n")

if __name__ == '__main__':
    # sourcemodel2target = {"gpt2-xl": ["gpt2", "gpt2-large"], "huggyllama/llama-13b": [ "huggyllama/llama-7b"], "bigscience/bloom-3b": ["bigscience/bloom-560m"], "facebook/opt-1.3b": ["facebook/opt-350m", "facebook/opt-125m"], "/checkpoint/swj0419/llama/30B": ["/checkpoint/swj0419/llama/7B"]} # "/checkpoint/swj0419/llama/13B"
    # sourcemodel2target = {"huggyllama/llama-13b": ["huggyllama/llama-7b"]} # , "huggyllama/llama-13b": ["huggyllama/llama-7b"]
    # sourcemodel2target = {"huggyllama/llama-13b": ["huggyllama/llama-7b", "gpt2", "gpt2-large"], "huggyllama/llama-30b": ["huggyllama/llama-7b","huggyllama/llama-13b", "gpt2-large", "gpt2"], "huggyllama/llama-65b": ["huggyllama/llama-7b", "huggyllama/llama-13b", "huggyllama/llama-30b", "gpt2", "gpt2-large"]} # , "gpt2-xl": ["gpt2"]
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
            for col_name in ["input", "paraphrase", "drop_0.1",  "drop_0.3",  "drop_0.5"]: # , "paraphrase", "drop_0.1",  "drop_0.3",  "drop_0.5" 
                for data_name in ["final_128"]: # "final_64", "final_32", 
                    # output path
                    output_dir = f"output/{source_name}/{data_name}/{name_part1}/{name_part2}/{col_name}"
                    print("output_dir: ", output_dir)
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    # input path
                    # input_path = f"/fsx-instruct-opt/swj0419/attack/wikidata/{source_name}_data/{data_name}.jsonl"
                    input_path = f"{data_name}.jsonl"
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

        
    