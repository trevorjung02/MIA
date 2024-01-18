from datasets import load_dataset
import pandas as pd
import json

CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
dataset = load_dataset("sentiment140", split='train', cache_dir=CACHE_DIR)

train_data = dataset['text'][:2000] 
val_data = dataset['text'][300000:302000]

train_df = pd.DataFrame(train_data, columns=['input'])
train_df.insert(1, 'label', 1)

val_df = pd.DataFrame(val_data, columns=['input'])
val_df.insert(1, 'label', 0)

df_tiny = pd.concat([train_df, val_df], ignore_index=True) 
df_debug = pd.concat([train_df[:100], val_df[:100]], ignore_index=True) 

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def dump_df(df, path):
    json_s = df.to_json(orient='records')
    data_dict = json.loads(json_s)
    dump_jsonl(data_dict, path)

dump_df(df_tiny, '/mmfs1/gscratch/ark/tjung2/MIA/data/sentiment140_oracle_tiny.jsonl')

dump_df(df_debug, '/mmfs1/gscratch/ark/tjung2/MIA/data/sentiment140_oracle_debug.jsonl')