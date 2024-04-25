import os
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
from plots import fig_fpr_tpr
import pandas as pd
import json
from rouge_score import rouge_scorer
from rank_bm25 import BM25Okapi
from argparse import ArgumentParser
from tqdm import tqdm

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--col', default = '', type=str, help="MIA attack to sort samples by")
    parser.add_argument('--mia_paths', type=str, help="Path to file containing list of files containing mia results")
    parser.add_argument('--top_n', type=int, help="Number of candidates to consider.")
    args = parser.parse_args()
    return vars(args)

def get_scores(df, bm25, corpus):
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_scores = []
    best_references = []
    candidates = df['input'].tolist()
    for candidate in tqdm(candidates):
        print(candidate)
        tokenized_query = candidate.split(" ")
        closest = bm25.get_top_n(tokenized_query, corpus, n=10)
        score = 0
        best_ref = ''
        for reference in closest:
            # print(reference)
            cur_score = scorer.score(candidate, reference)['rougeL'][2]
            if cur_score > score:
                score = cur_score
                best_ref = reference
        rouge_scores.append(score)
        best_references.append(best_ref)
        print(score)
        print(best_ref)
        print('--------------------------')
    return rouge_scores, best_references
    
def main():
    args = get_cli_args()

    res_dir = create_new_dir('extraction')
    res_path = os.path.join(res_dir, 'rouge_scores.jsonl')
    args_path = os.path.join(res_dir, 'args.json')
    num_path = os.path.join(res_dir, 'num_extracted.txt')
    
    train_path = '/mmfs1/gscratch/zlab/swj0419/MIA/kNNLM_privacy/data/enron/train.txt'
    args['train_path'] = train_path
    with open(train_path) as f:
        corpus = [l.strip() for l in f]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    paths = []
    all_output = []
    for i in range(122, 133):
        if i==125:
            continue
        output_dir = f'/mmfs1/gscratch/zlab/swj0419/MIA/output/agnews/none/none/run_{i}'
        paths.append(output_dir)
        all_output.extend(read_jsonl(f"{output_dir}/output.jsonl"))
    args['paths'] = paths
    
    with open(args_path, 'w') as f:
        f.write(json.dumps(args))
    print(args) 

    df = pd.DataFrame(all_output)
    df_2 = pd.DataFrame(df.pred.tolist())
    df = df.join(df_2)

    df = df.sort_values(args['col'])[:args['top_n']]
    # ref_samples = df.sort_values('loss/ref_loss')[:5000]
    # rmia_mean_samples = df.sort_values("{'do_sample': True},prefix_len=[0.1] RMIA mean")[:5000]
    # rmia_samples = df.sort_values("{'do_sample': True},prefix_len=[0.1] RMIA(gamma=0.95)")[:5000]

    rouge_scores, best_references = get_scores(df, bm25, corpus)
    df_3 = df[['ip', 'i', 'input', 'loss']]
    df_3.insert(len(df_3.columns), 'rougeL', rouge_scores)
    df_3.insert(len(df_3.columns), 'best ref', best_references)

    with open(res_path, 'w') as f:
        for r in json.loads(df_3.to_json(orient='records')):
            # print(r)
            f.write(json.dumps(r) + '\n')
        
    num_extracted = len([x for x in rouge_scores if x >= 0.5])
    print(num_extracted)
    with open(num_path, 'w') as f:
        f.write(f"{num_extracted}")

if __name__ == "__main__":
    main()