import os
os.chdir('/gscratch/zlab/swj0419/MIA/src/MIA')
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
import pandas as pd
import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pylcs

def add_distances(df, corpus, bm25, n):
    metrics = ['rougeL', 'rouge1', 'rouge2', 'lcs', 'edit distance'] 
    scores = {d: [] for d in metrics} 
    closest_refs = {d: [] for d in metrics} 
    for i in tqdm(range(len(df))):
        r = df.iloc[i]
        if r['label'] == 1:
            for d in metrics:
                scores[d].append(get_score_positive(d))
                closest_refs[d].append(None)
        else:
            q = r['input']
            tokenized_query = q.split(" ")
            closest = bm25.get_top_n(tokenized_query, corpus, n=n)
            for d in metrics:
                score, ref = get_score_and_ref(q, closest, d)
                scores[d].append(score)
                closest_refs[d].append(ref)
    for d in metrics:
        df[f'{d} score'] = scores[d]
        df[f'{d} closest'] = closest_refs[d]
    return scores, closest_refs

def get_score_positive(d):
    if 'rouge' in d or d == 'lcs':
        score = 1
    elif d == 'edit distance':
        score = 0
    return score 

def get_score_and_ref(q, closest, d):
    best_ref = None
    if d == 'rougeL':
        scorer = rouge_scorer.RougeScorer(['rougeL'])
        score = 0
        for reference in closest:
            # print(reference)
            cur_score = scorer.score(q, reference)['rougeL'][2]
            if cur_score > score:
                score = cur_score
                best_ref = reference
    elif d == 'rouge1':
        scorer = rouge_scorer.RougeScorer(['rouge1'])
        score = 0
        for reference in closest:
            # print(reference)
            cur_score = scorer.score(q, reference)['rouge1'][2]
            if cur_score > score:
                score = cur_score
                best_ref = reference
    elif d == 'rouge2':
        scorer = rouge_scorer.RougeScorer(['rouge2'])
        score = 0
        for reference in closest:
            # print(reference)
            cur_score = scorer.score(q, reference)['rouge2'][2]
            if cur_score > score:
                score = cur_score
                best_ref = reference
    elif d == 'lcs':
        score = 0
        for reference in closest:
            # print(reference)
            cur_score = pylcs.lcs_sequence_length(q, reference) / len(q)
            if cur_score > score:
                score = cur_score
                best_ref = reference
    elif d == 'edit distance':
        score = 1
        for reference in closest:
            # print(reference)
            cur_score = pylcs.edit_distance(q, reference) / len(q)
            if cur_score < score:
                score = cur_score
                best_ref = reference
    return score, best_ref

def get_semantic_scores(queries, corpus):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = corpus_embeddings.to("cuda")
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)
    query_embeddings = query_embeddings.to("cuda")
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
    return hits

def main():
    os.chdir('/gscratch/zlab/swj0419/MIA')
    res_dir = create_new_dir('scores')
    res_path = os.path.join(res_dir, 'df.csv')
    
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
    
    add_distances(df, corpus, bm25, 100)

    hits = get_semantic_scores(df['input'].tolist(), corpus)
    df['semantic score'] = [hit[0]['score'] for hit in hits]
    df['semantic closest'] = [corpus[hit[0]['corpus_id']] for hit in hits] 
    
    with open(res_path, 'w') as f:
        df.to_csv(f)

if __name__ == "__main__":
    main()