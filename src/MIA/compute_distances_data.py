import os
import pandas as pd
import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pylcs
import numpy as np
import editdistance
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def create_new_dir(dir: str) -> str:
    """Automatically create directory for saving result of run with path 'dir/run_#'"""
    if not os.path.exists(dir):
        num = 1
    else:
        files = set(os.listdir(dir))
        num = len(files)+1
        while f"run_{num}" in files:
            num += 1
    new_dir = os.path.join(dir, f"run_{num}")
    done = False
    while not done:
        try:
            os.makedirs(new_dir, exist_ok=False)
        except: 
            num += 1
            new_dir = os.path.join(dir, f"run_{num}")
        else:
            done = True
    return new_dir

def read_jsonl(path):
    """Read jsonl file"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def add_distances(df, corpus, n):
    """Add syntactic distance metrics to dataframe
    
    Arguments:
    df: dataframe of mia evaluation data
    corpus: training data
    n: number of training data used to compute distance metric, taken from bm25
    """
    
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # metrics = ['rougeL', 'rouge1', 'rouge2', 'lcs', 'edit distance'] 
    metrics = ['edit distance'] 
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
    """Get distance score for a member data"""
    if 'rouge' in d or d == 'lcs':
        score = 1
    elif d == 'edit distance':
        score = 0
    return score 

def get_score_and_ref(q, closest, d):
    """Get distance score and closest training data for query"""
    best_ref = None
    if d in {'rougeL', 'rouge1', 'rouge2'}:
        scorer = rouge_scorer.RougeScorer([d])
        score = 0
        for reference in closest:
            # print(reference)
            cur_score = scorer.score(q, reference)[d][2]
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
        q_ids = tokenizer(q)['input_ids']
        score = np.inf
        for reference in closest:
            # print(reference)
            reference_ids = tokenizer(reference)['input_ids']
            cur_score = editdistance.eval(q_ids, reference_ids) / len(q_ids)
            if cur_score < score:
                score = cur_score
                best_ref = reference
    return score, best_ref

def get_semantic_scores(queries, corpus):
    """Get semantic distance scores and closest training data"""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = corpus_embeddings.to("cuda")
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)
    query_embeddings = query_embeddings.to("cuda")
    query_embeddings = util.normalize_embeddings(query_embeddings)
    
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
    scores = [hit[0]['score'] for hit in hits]
    references = [corpus[hit[0]['corpus_id']] for hit in hits] 
    return scores, references

def main():
    res_dir = create_new_dir('/mmfs1/gscratch/zlab/swj0419/MIA/scores')
    res_path = os.path.join(res_dir, 'df.csv')

    # Path to training data
    train_path = '/mmfs1/gscratch/zlab/swj0419/MIA/data/newsSpace_oracle_target_train.csv'
    corpus_df = pd.read_csv(train_path)
    corpus = corpus_df['description'].tolist()

    # Path to mia evaluation data
    mia_dataset_path = '/mmfs1/gscratch/zlab/swj0419/MIA/data/newsSpace_oracle_1000.jsonl'
    output = read_jsonl(mia_dataset_path)
    
    df = pd.DataFrame(output)
    
    add_distances(df, corpus, 100)

    scores, refs = get_semantic_scores(df['input'].tolist(), corpus)
    df['semantic score'] = scores
    df['semantic closest'] = refs
    
    with open(res_path, 'w') as f:
        df.to_csv(f)

if __name__ == "__main__":
    main()