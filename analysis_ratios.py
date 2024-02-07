import token
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import math
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import statistics
import numpy as np
import logging
import random
import io
from src.MIA.utils import evaluate_model, unzip_collate
from src.MIA.neighbor_sampling import sample_generation, get_indices_individual

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.3f}'.format})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=CACHE_DIR)
# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
target_model.config.pad_token_id = target_model.config.eos_token_id
target_model.generation_config.pad_token_id = target_model.config.eos_token_id
path = "/mmfs1/gscratch/ark/tjung2/MIA/checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533"
ckpt = torch.load(path, map_location=device)
target_model.load_state_dict(ckpt['model_state_dict'])
target_model.eval()
target_model.to(device)

ref_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=CACHE_DIR)
# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
ref_model.config.pad_token_id = ref_model.config.eos_token_id
ref_model.generation_config.pad_token_id = ref_model.config.eos_token_id
ref_model.eval()
ref_model.to(device)

path = '/mmfs1/gscratch/ark/tjung2/MIA/output/agnews/epoch-3_perplexity-20.9533/none/run_285/output.jsonl'
with open(path) as f:
    df = pd.read_json(f, lines=True)

idx = 31
sentence = df['input'].iloc[idx]
print(sentence)

test_dl = DataLoader([sentence], batch_size=32, shuffle=False)
target_losses, target_variances, unreduced_losses, _, _ = evaluate_model(target_model, tokenizer, test_dl) 
ref_losses, ref_variances, unreduced_losses_ref, _, _ = evaluate_model(ref_model, tokenizer, test_dl)

token_ratios = unreduced_losses[0][0] / unreduced_losses_ref[0][0]
token_ratios = token_ratios.to('cpu')

l = tokenizer.batch_decode(tokenizer(sentence, return_tensors='pt')['input_ids'][0][np.argsort(token_ratios)+1])
print(f"Tokens sorted by target_loss / ref_loss::\n{l}")


values, indices = torch.sort((torch.sum(unreduced_losses[0][0]) - unreduced_losses[0][0]) / (torch.sum(unreduced_losses_ref[0][0]) - unreduced_losses_ref[0][0]), descending=True)
l = tokenizer.batch_decode(tokenizer(sentence, return_tensors='pt')['input_ids'][0][indices.to('cpu')+1])
print(f"Tokens sorted by weighted target_loss / ref_loss:\n{l}")

print(f"Sorted target/ref loss:\n{np.sort(token_ratios)}")

sorted_target_losses = unreduced_losses[0][0][torch.argsort(token_ratios)].cpu().numpy()
print(f"target loss sorted by target/ref loss:\n{sorted_target_losses}")


