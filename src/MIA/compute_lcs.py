import os
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
from plots import fig_fpr_tpr
import pandas as pd
import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pylcs

train_path = '/mmfs1/gscratch/zlab/swj0419/MIA/data/newsSpace_oracle_target_train.csv'
corpus_df = pd.read_csv(train_path)
corpus = corpus_df['description'].tolist()
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

all_output = []
for i in [22]:
    output_dir = f'/mmfs1/gscratch/zlab/swj0419/MIA/output/agnews/none/none/run_{i}'
    all_output.extend(read_jsonl(f"{output_dir}/output.jsonl"))

df = pd.DataFrame(all_output)
df_2 = pd.DataFrame(df.pred.tolist())
df = df.join(df_2)

def get_lcs_scores(queries, corpus, bm25):
    lcs_scores = []
    best_references = []
    for q in tqdm(queries):
        tokenized_query = q.split(" ")
        closest = bm25.get_top_n(tokenized_query, corpus, n=1)
        score = 0
        best_ref = ''
        lcs_lengths = pylcs.lcs_sequence_of_list(q, closest)
        for i in range(len(closest)):
            # print(reference)
            reference = closest[i]
            cur_score = lcs_lengths[i]
            if cur_score > score:
                score = cur_score
                best_ref = reference
        lcs_scores.append(score)
        best_references.append(best_ref)
    return lcs_scores, best_references

df_out = df[df['label'] == 0]
lcs_scores, best_refs = get_lcs_scores(df_out['input'].tolist(), corpus, bm25)

with open('/mmfs1/gscratch/zlab/swj0419/MIA/scores/lcs.json', 'w') as f:
    f.write(json.dumps({'lcs_scores': lcs_scores, 'best_ref': best_refs}))