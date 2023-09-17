import pandas as pd
import json
from tqdm import tqdm

val_path = '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_val.csv'
val_df = pd.read_csv(val_path, names=['id', 'category', 'input'], header=0)
val_df.insert(2, 'label', 0)

train_path = '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_target_train.csv'
train_df = pd.read_csv(train_path, names=['id', 'category', 'input'], header=0)
train_df.insert(2, 'label', 1)

df = pd.concat([train_df, val_df], ignore_index=True)
df = df.sample(frac=1)

df_tiny = pd.concat([train_df[:2000], val_df[:2000]], ignore_index=True)
df_tiny = df_tiny.sample(frac=1)

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

json_s = df.to_json(orient='records')
data_dict = json.loads(json_s)
dump_jsonl(data_dict, '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle.jsonl')

json_s_tiny = df_tiny.to_json(orient='records')
data_dict_tiny = json.loads(json_s_tiny)
dump_jsonl(data_dict_tiny, '/mmfs1/gscratch/ark/tjung2/MIA/data/newsSpace_oracle_tiny.jsonl')
