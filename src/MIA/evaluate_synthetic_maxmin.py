import os
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
from plots import fig_fpr_tpr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plots import fig_fpr_tpr
import json
import re
from tqdm import tqdm
import random

distance_metric = "edit"

def compute_auc(name, member_run, nonmember_run):
    # Match synthetic data to original data
    member_dir = f'/mmfs1/gscratch/zlab/swj0419/MIA/output/agnews/none/none/run_{member_run}'
    df_members = pd.DataFrame(read_jsonl(f'{member_dir}/output.jsonl'))
    nonmember_dir = f'/mmfs1/gscratch/zlab/swj0419/MIA/output/agnews/none/none/run_{nonmember_run}'
    df_nonmembers = pd.DataFrame(read_jsonl(f'{nonmember_dir}/output.jsonl'))
    df_mia = pd.concat([df_members, df_nonmembers])
    input_to_syn_indices = {}
    for i in range(len(df_mia)):
        input = df_mia.iloc[i]['input']
        if input not in input_to_syn_indices:
            input_to_syn_indices[input] = []
        input_to_syn_indices[input].append(i)

    def update_col_names(df):
        names = dict()
        for column in pd.DataFrame(df['pred'].tolist()).columns: 
            if "neighbor(gamma=" in column:
                gamma = float(re.findall("gamma=(\d+.\d+)", column)[0])
                gamma = str(np.round(gamma,2))
                names[column] = "{'do_sample': True},prefix_len=[0.1] neighbor(gamma=" + gamma + ")"
        
        df_pred = pd.DataFrame(df['pred'].tolist())
        df_pred = df_pred.rename(columns=names)
        df['pred'] = json.loads(df_pred.to_json(orient='records'))

    def update_agg_scores(df_ori, agg):
        if agg == "max":
            agg_f = max
        elif agg == "min":
            agg_f = min
        
        for i in tqdm(range(len(df_ori))):
            r = df_ori.iloc[i]
            if agg == "max|min":
                if r['label'] == 1:
                    agg_f = max
                else:
                    agg_f = min
                    
            neighbor_indices = input_to_syn_indices[r['input']]
            agg_scores = {}
            for col in r['pred']:
                agg_score = r['pred'][col]
                neighbors_have_col = True
                for neighbor_idx in neighbor_indices:
                    if col in df_mia.iloc[neighbor_idx]['pred']:
                        agg_score = agg_f(agg_score, df_mia.iloc[neighbor_idx]['pred'][col])
                    else:
                        neighbors_have_col = False
                        break
                if neighbors_have_col:
                    agg_scores[f"{col} {agg}"] = agg_score
            r['pred'].update(agg_scores)

    
    # Take max/min of mia scores
    for agg in ['max', 'min', 'max|min']:
        ori_dir = f'/mmfs1/gscratch/zlab/swj0419/MIA/output/agnews/none/none/run_{36}'
        df_ori = pd.DataFrame(read_jsonl(f'{ori_dir}/output.jsonl'))
        
        update_col_names(df_ori)
        update_agg_scores(df_ori, agg)
        
        # Filter the MIA dataset to remove nonmembers in the neighborhood of members, and balance the classes
        df_scores = pd.read_csv('scores/run_15/df.csv')
        df_scores_out = df_scores[df_scores['label']==0]
        sentences_to_remove = set(df_scores_out[df_scores_out['edit distance score'] < 0.1]['input'].tolist())
        pos_indices = df_scores[df_scores['label']==1].index.tolist()
        # pos_indices_to_remove = random.sample(pos_indices, len(sentences_to_remove))
        pos_indices_to_remove = pos_indices[:len(sentences_to_remove)]
        sentences_to_remove.update(set(df_scores.iloc[pos_indices_to_remove]['input'].tolist()))
        df_filtered = df_ori[~df_ori['input'].isin(sentences_to_remove)]
        
        # Calculate AUC
        json_s = df_filtered.to_json(orient='records')
        data_dict = json.loads(json_s)
        out_dir = create_new_dir(f'/mmfs1/gscratch/zlab/swj0419/MIA/temp/{name}_{agg}')
        dump_jsonl(data_dict, f"{out_dir}/output.jsonl")
        
        fig_fpr_tpr(data_dict, out_dir)

def main():
    runs = {
        'edit_0.05_multinomial_percent': [323, 325],
        'edit_5_random_percent': [320, 321],
        'edit_1': [322, 326],
        'edit_5': [324, 327],
        'edit_10': [329, 333],
        'edit_10_random': [328, 331],
        'edit_5_random': [330, 335],
        'edit_1_random': [332, 334],
    }
    for name, (member_run, nonmember_run) in runs.items():
        compute_auc(name, member_run, nonmember_run)

if __name__ == "__main__":
    main()