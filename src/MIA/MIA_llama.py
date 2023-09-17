import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
from pprint import pprint
import torch
import zlib
import json
from tqdm import tqdm
import pandas as pd
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve

from plots import fig_fpr_tpr
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir
from detectGPT import DetectGPTPerturbation
from torch.utils.data import DataLoader

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


def inference(model1, model2, tokenizer1, tokenizer2, text, ex, losses1, losses2, args):
    # ex["pred"] = {}
    pred = {}

    # perplexity of large and small models
    p_target, all_prob = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_ref, _ = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
    p_lower, _= calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    # detectGPT 
    # perturbed_text = detect_gpt.perturb(text)
    # p_perturb, _ = calculatePerplexity(perturbed_text, model1, tokenizer1, gpu=model1.device)

    # detectGPT on reference
    # p_ref_perturb, _ = calculatePerplexity(perturbed_text, model2, tokenizer2, gpu=model2.device)

    # ppl
    pred["ppl"] = p_target
    # Ratio of log ppl of large and small models
    pred["log_ppl/log_ref_ppl"] = (np.log(p_target)/np.log(p_ref)).item()
    
    # pred[f"log_ppl/log_ppl_perturb"] = (np.log(p_target)/np.log(p_perturb)).item()

    num_z_dominated = 0
    for loss1, loss2 in zip(losses1, losses2):
        if np.log(p_target) / loss1 * loss2 / np.log(p_ref) < args['gamma']:
            num_z_dominated += 1
    pred["RMIA"] = 1 - num_z_dominated / len(losses1)
    print(pred['RMIA'])

    # Ratio of log ppl of lower-case and normal-case
    pred["log_ppl/log_lower_ppl"] = (np.log(p_target) / np.log(p_lower)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["log_ppl/zlib"] = np.log(p_target)/zlib_entropy
    # token mean and var
    # pred["mean"] = -np.mean(all_prob).item()
    # pred["var"] = np.var(all_prob).item()
    
    ex["pred"] = pred
    return ex

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

def evaluate_data(test_data, rmia_data, model1, model2, tokenizer1, tokenizer2, col_name, args):
    rmia_dl = DataLoader(rmia_data, batch_size=32, shuffle=False)
    losses1 = evaluate_model(model1, tokenizer1, rmia_dl)
    losses2 = evaluate_model(model2, tokenizer2, rmia_dl)

    print(f'losses1[:10] = {losses1[:10]}')
    print(f'len(losses1) = {len(losses1)}')
    print(f'losses2[:10] = {losses2[:10]}')
    print(f'len(losses2) = {len(losses2)}')
    all_output = []
    for ex in tqdm(test_data): # [:100]
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex, losses1, losses2, args)
        all_output.append(new_ex)
    return all_output

def get_name(path):
    if path:
        return path.split('/')[-1]
    return 'none'

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--num_z', type=int, default = 100)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = get_cli_args()

    model1, model2, tokenizer1, tokenizer2 = load_model('gpt2', 'gpt2', target_path=args['target_path'], ref_path=args['ref_path'])
    
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
    data = load_jsonl(input_path)

    # RMIA data
    rmia_path = 'data/newsSpace_oracle_val.csv'
    args['rmia_path'] = rmia_path
    df = pd.read_csv(rmia_path, names=['id', 'category', 'text'], header=0, index_col='id', encoding='utf-8')
    rmia_data = df['text'].astype(str).to_list()[:args['num_z']]

    with open(output_dir + '/args.json', mode='w') as f:
        json.dump(args, f)

    all_output = evaluate_data(data, rmia_data, model1, model2, tokenizer1, tokenizer2, 'input', args)
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
