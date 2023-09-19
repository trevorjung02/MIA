import logging
logging.basicConfig(level='ERROR')
import torch
import zlib
import json
import io
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve

from plots import fig_fpr_tpr
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir
from detectGPT import DetectGPTPerturbation

# load detectGPT
# detect_gpt = DetectGPTPerturbation()

# Returns the perplexity of the batch (in this case, just one sentence), and the list of the probabilities the model gives to each labeled token
def calculatePerplexity(sentence, model, tokenizer, gpu):

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob

def RMIA_score(gamma, target_loss, target_ref, rmia_losses):
    threshold = gamma / target_loss * target_ref 
    num_z_dominated = torch.searchsorted(rmia_losses, threshold).item()
    return 1 - num_z_dominated / len(rmia_losses)

def compute_decision(target_loss, target_ref, target_losses_z, ref_losses_z, args):
    pred = {}

    # Loss
    pred["loss"] = target_loss.item()
    # Ratio of loss of large and small models
    pred["loss/ref_loss"] = (target_loss/target_ref).item()
    
    # RMIA
    rmia_losses = ref_losses_z / target_losses_z 
    rmia_losses = torch.sort(rmia_losses)[0]

    if args['gamma'] == -1:
        for gamma in np.arange(0.5, 1.35, 0.05):
            pred[f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, target_ref, rmia_losses)
    else:
        gamma = args['gamma']
        pred[f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, target_ref, rmia_losses)
    return pred

def evaluate_model(model, tokenizer, dl):
    val_bar = tqdm(range(len(dl)))
    losses = []
    for batch in dl:
        batch = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=150)
        labels = torch.tensor([
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch['attention_mask'], batch['input_ids'])]
            ])
        batch['labels'] = labels
        batch = batch.to(model.device)
        
        with torch.no_grad():
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.transpose(1,2), shift_labels)
            num_tokens = torch.sum(shift_labels != -100, dim=1)
            loss_sum = torch.sum(loss, dim=1)
            loss = loss_sum / num_tokens
            # print(loss)
            losses.append(loss)
        val_bar.update(1)
    losses = torch.cat(losses)
    # with torch.no_grad():
    #     outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
    #     losses.append(outputs.loss)
    return losses

def jsonl_to_dl(test_data):
    test_json = json.dumps(test_data)
    test_df = pd.read_json(io.StringIO(test_json))['input']
    test_l = test_df.astype(str).tolist()
    return DataLoader(test_l, batch_size=32, shuffle=False)

def mia(test_data, rmia_data, target_model, ref_model, tokenizer1, tokenizer2, args):
    rmia_dl = DataLoader(rmia_data, batch_size=32, shuffle=False)
    target_losses_z = evaluate_model(target_model, tokenizer1, rmia_dl)
    ref_losses_z = evaluate_model(ref_model, tokenizer2, rmia_dl)

    test_dl = jsonl_to_dl(test_data) 
    target_losses = evaluate_model(target_model, tokenizer1, test_dl)
    ref_losses = evaluate_model(ref_model, tokenizer2, test_dl)

    all_output = []
    for i in tqdm(range(len(test_data))): # [:100]
        pred = compute_decision(target_losses[i], ref_losses[i], target_losses_z, ref_losses_z, args)
        new_ex = test_data[i].copy()
        new_ex['pred'] = pred
        all_output.append(new_ex)
    return all_output

def get_name(path):
    if path:
        return path.split('/')[-1]
    return 'none'

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--num_z', type=int, default = 6000)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = get_cli_args()

    target_model, ref_model, tokenizer1, tokenizer2 = load_model('gpt2', 'gpt2', target_path=args['target_path'], ref_path=args['ref_path'])
    
    # output path
    name1 = get_name(args['target_path'])
    name2 = get_name(args['ref_path'])
    output_dir = f"output/agnews/{name1}/{name2}"
    output_dir = create_new_dir(output_dir)
    args['output_dir'] = output_dir
    print("output_dir: ", output_dir)

    # input path
    input_path = f"data/newsSpace_oracle_tiny.jsonl"
    args['input_path'] = input_path
    test_data = load_jsonl(input_path)

    # RMIA data
    rmia_path = 'data/newsSpace_oracle_val.csv'
    args['rmia_path'] = rmia_path
    rmia_data = pd.read_csv(rmia_path, names=['id', 'category', 'text'], header=0,  encoding='utf-8')['text'][:args['num_z']]
    rmia_data = rmia_data.astype(str).tolist()

    with open(output_dir + '/args.json', mode='w') as f:
        json.dump(args, f)

    all_output = mia(test_data, rmia_data, target_model, ref_model, tokenizer1, tokenizer2, args)
    '''
    dump and read data 
    '''
    dump_jsonl(all_output, f"{output_dir}/output.jsonl")
    all_output = read_jsonl(f"{output_dir}/output.jsonl")
    # bp()
    '''
    plot
    '''
    fig_fpr_tpr(all_output, output_dir)
    # compute_topkmean(all_output, output_dir)
    print("===============================")
