import pandas as pd
import json
from tqdm import tqdm

def load_csv(path, label):
    df = pd.read_csv(path, names=['id', 'category', 'input'], header=0)
    df.insert(2, 'label', label)
    return df

def create_df(n, train_df, val_df):
    if n < 0:
        df = pd.concat([train_df, val_df], ignore_index=True) 
    else:
        df = pd.concat([train_df[:n], val_df[:n]], ignore_index=True)
    df = df.sample(frac=1)
    return df

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def dump_df(df, path):
    json_s = df.to_json(orient='records')
    data_dict = json.loads(json_s)
    dump_jsonl(data_dict, path)

def main():
    val_path = '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_val.csv'
    val_df = load_csv(val_path, 0)

    train_path = '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_target_train.csv'
    train_df = load_csv(train_path, 1)

    df = create_df(-1, train_df, val_df)

    df_tiny = create_df(2000, train_df, val_df)

    df_debug = create_df(100, train_df, val_df)
    
    dump_df(df, '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle.jsonl')

    dump_df(df_tiny, '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_tiny.jsonl')

    dump_df(df_debug, '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_debug.jsonl')

if __name__ == "__main__":
    main()