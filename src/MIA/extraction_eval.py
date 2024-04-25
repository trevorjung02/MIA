import os
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
import pandas as pd
import json
from rouge_score import rouge_scorer
from argparse import ArgumentParser
from tqdm import tqdm

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--col', default = '', type=str, help="MIA attack to sort samples by")
    parser.add_argument('--mia_paths', type=str, help="Path to file containing list of files containing mia results")
    args = parser.parse_args()
    return vars(args)

def main():
    args = get_cli_args()

    res_dir = create_new_dir('extraction')
    res_path = os.path.join(res_dir, 'rouge_scores.jsonl')
    args_path = os.path.join(res_dir, 'args.json')
    num_path = os.path.join(res_dir, 'num_extracted.txt')
    
    train_path = '/mmfs1/gscratch/zlab/swj0419/MIA/data/newsSpace_oracle_target_train.csv'
    df_train = pd.read_csv(train_path)
    train_data = df_train['description'].tolist()

    paths = []
    with open(args['mia_paths']) as f:
        for l in f:
            paths.append(l.rstrip('\n'))
    args['paths'] = paths
    args['res_dir'] = res_dir

    with open(args_path, 'w') as f:
        f.write(json.dumps(args))
    print(args) 

    df = get_mia_scores(paths)

    if len(args['col']) > 0:
        df = df.sort_values(args['col'])[:1000]

    rouge_scores, originals = get_scores(df, train_data)
    df.insert(len(df.columns), 'rougeL', rouge_scores)
    df.insert(len(df.columns), 'original', originals)
    # ref_samples = df.sort_values('loss/ref_loss')[:5000]
    # rmia_mean_samples = df.sort_values("{'do_sample': True},prefix_len=[0.1] RMIA mean")[:5000]

    with open(res_path, 'w') as f:
        for r in json.loads(df.to_json(orient='records')):
            # print(r)
            f.write(json.dumps(r) + '\n')

    num_extracted = len([x for x in rouge_scores if x >= 0.5])
    print(num_extracted)

    with open(num_path, 'w') as f:
        f.write(num_extracted)

def get_scores(df, train_data):
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_scores = []
    originals = []
    for candidate in tqdm(df['input'].tolist()):
        print(candidate)
        score = 0
        original = None
        for reference in train_data:
            # print(reference)
            scores = scorer.score(candidate, reference)
            # print(scores)
            if scores['rougeL'][2] > score:
                score = scores['rougeL'][2]
                original = reference
        rouge_scores.append(score)
        originals.append(original)
        print(score)
        print(original)
        print("-----")
    return rouge_scores, originals

def get_mia_scores(paths):
    all_output = []
    for output_dir in paths:
        all_output.extend(read_jsonl(f"{output_dir}/output.jsonl"))
    df = pd.DataFrame(all_output)
    df_2 = pd.DataFrame(df.pred.tolist())
    df = df.join(df_2)
    return df

if __name__ == '__main__':
    main()

