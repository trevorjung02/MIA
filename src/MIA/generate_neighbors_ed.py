import os
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from tqdm import tqdm
import evaluate
import editdistance
import math
from sentence_transformers import SentenceTransformer, util

def evaluate_neighbor_one(query, target, rouge, model):
    rouge_score = rouge.compute(predictions=query, references=target, use_aggregator=False) 
    pred_embeddings = model.encode(query, convert_to_tensor=True)
    ref_embeddings = model.encode(target, convert_to_tensor=True)
    cos_sim = util.cos_sim(pred_embeddings, ref_embeddings).cpu().numpy().squeeze().tolist()
    rougel = rouge_score["rougeL"][0]
    # print("Rouge score: ", rougel)
    # print("Cosine similarity: ", cos_sim)
    return rougel, cos_sim

@torch.no_grad
def generate_synthetic(data, num_z_per_data, num_edits, model, sim_model, rouge, tokenizer, method, percent, label):
    synthetic_data = []
    num_incorrect_dist = 0
    total_distance_off = 0
    for s in tqdm(data):
        ids = tokenizer(s, return_tensors='pt')['input_ids'][0].to(model.device)
        logits = model(ids).logits[:-1]
        
        if method == "multinomial":    
            for i in range(len(logits)):
                logits[i][ids[i+1]] = -float("Inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        for _ in range(num_z_per_data):
            if percent:
                cur_num_edits = math.ceil(num_edits * len(logits))
            else:
                cur_num_edits = min(num_edits, len(logits))
            # print(f"cur_num_edits = {cur_num_edits}")
            indices = random.sample(range(len(logits)), cur_num_edits)
            # print(f"indices = {indices}")
            cur_ids = torch.clone(ids)
            for i in indices:
                # ids[i] = random.randint(0, tokenizer.vocab_size-1)
                if method == "multinomial":
                    cur_ids[i+1] = torch.multinomial(probs[i], 1, replacement=True)[0]
                else:
                    cur_ids[i+1] = torch.randint(len(logits[i]), (1,))[0]
            syn = tokenizer.decode(cur_ids)

            # Check edit distance
            syn_ids = tokenizer(syn)['input_ids']
            edit_dist = editdistance.eval(ids.tolist(), syn_ids)
            if edit_dist != cur_num_edits:
                num_incorrect_dist += 1
                total_distance_off += abs(edit_dist-cur_num_edits)
            #     print(f"edit_dist = {edit_dist}")
            #     print(f"cur_num_edits = {cur_num_edits}")
            #     print(f"ori_ids = {ids.tolist()}")
            #     print(f"syn_ids = {syn_ids}")
            #     print(f"indices = {indices}")
            
            rougel, cos_sim = evaluate_neighbor_one([syn], [s], rouge, sim_model)
            example = {"input": s, "syn": syn, "rouge": rougel, "cosine": cos_sim, "label": label}
            synthetic_data.append(example)
    print(f"Number of data with incorrect edit distance = {num_incorrect_dist}")
    print(f"Average incorrect edit distance = {total_distance_off / num_incorrect_dist}")
    return synthetic_data


def main():
    model = AutoModelForCausalLM.from_pretrained('gpt2').cuda()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    rouge = evaluate.load('rouge')
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    data = read_jsonl('data/newsSpace_oracle_1000.jsonl')
    df = pd.DataFrame(data)
    df_pos = df[df['label']==1]
    df_neg = df[df['label']==0]
    members = df_pos['input'].tolist()
    nonmembers = df_neg['input'].tolist()

    output_dir = '/gscratch/zlab/swj0419/MIA/data/edit_distance'
    for method in ["multinomial", "random"]:
        for num_neighbors in [1, 10]:
            percent = False
            for num_edits in [1,5, 10]:
                print(f"newsSpace_edit={num_edits}_{method}_k={num_neighbors}")
                synthetic_data = generate_synthetic(members, num_neighbors, num_edits, model, sim_model, rouge, tokenizer, method, percent, 1)
                synthetic_data.extend(generate_synthetic(nonmembers, num_neighbors, num_edits, model, sim_model, rouge, tokenizer, method, percent, 0))
                dump_jsonl(synthetic_data, f'{output_dir}/newsSpace_edit={num_edits}_{method}_k={num_neighbors}.jsonl')
        
            # Percent datasets
            percent = True
            for percent_edits in [0.05, 0.1]:
                print(f"newsSpace_edit={percent_edits}percent_{method}_k={num_neighbors}")
                synthetic_data = generate_synthetic(members, num_neighbors, percent_edits, model, sim_model, rouge, tokenizer, method, percent, 1)
                synthetic_data.extend(generate_synthetic(nonmembers, num_neighbors, percent_edits, model, sim_model, rouge, tokenizer, method, percent, 0))
                dump_jsonl(synthetic_data, f'{output_dir}/newsSpace_edit={percent_edits}percent_{method}_k={num_neighbors}.jsonl')

if __name__ == "__main__":
    main()