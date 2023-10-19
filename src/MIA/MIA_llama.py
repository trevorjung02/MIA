import logging
logging.basicConfig(level='ERROR')
import torch
import zlib
import json
import io
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve
from transformers import BertForMaskedLM, BertTokenizer
from scipy.stats import norm

from plots import fig_fpr_tpr
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir
from neighbor_sampling import generate_neighbors, sample_generation

# load detectGPT
# detect_gpt = DetectGPTPerturbation()

# Returns the perplexity of the batch (in this case, just one sentence), and the list of the probabilities the model gives to each labeled token
def calculatePerplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
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

def RMIA_score(gamma, target_loss, ref_loss, rmia_losses):
    threshold = gamma / target_loss * ref_loss 
    num_z_dominated = torch.searchsorted(rmia_losses, threshold).item()
    return 1 - num_z_dominated / len(rmia_losses)

def evaluate_RMIA(text, target_loss, ref_loss, target_model, ref_model, tokenizer1, tokenizer2, args):
    res = {}
    name = f"{args['generate_args']},prefix_len={args['prefix_length']} "
    neighbors = sample_generation(text, target_model, tokenizer1, args)
    # neighbors = [text for text, cand, p in neighbor_tuples]
    # print(f"neighbors = {neighbors}")
    
    rmia_dl = DataLoader(neighbors, batch_size=32, shuffle=False)
    target_losses_z = evaluate_model(target_model, tokenizer1, rmia_dl)
    ref_losses_z = evaluate_model(ref_model, tokenizer2, rmia_dl)

    ratio = torch.count_nonzero(target_loss < target_losses_z).item() / len(target_losses_z)
    res[name + 'neighbor'] = 1-ratio

    res[name + 'loss/neighbors'] = (target_loss / torch.mean(target_losses_z)).item()

    res[name + 'loss neighbor gaussians'] = norm.cdf(target_loss.item(), loc=torch.mean(target_losses_z).item(), scale=torch.std(target_losses_z).item())

    # RMIA
    rmia_losses = ref_losses_z / target_losses_z 
    rmia_losses = torch.sort(rmia_losses)[0]

    if args['gamma'] == -1:
        for gamma in np.arange(0.5, 1.35, 0.05):
            res[name + f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, ref_loss, rmia_losses)
    else:
        gamma = args['gamma']
        res[name + f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, ref_loss, rmia_losses)

    res[name + 'RMIA mean'] = (target_loss / ref_loss * torch.mean(rmia_losses)).item()
    return res

def compute_decision_offline(target_loss, ref_loss, target_losses_z, ref_losses_z, args):
    pred = {}

    # Loss
    pred["loss"] = target_loss.item()
    # Ratio of loss of large and small models
    pred["loss/ref_loss"] = (target_loss/ref_loss).item()
    # RMIA
    rmia_losses = ref_losses_z / target_losses_z 
    rmia_losses = torch.sort(rmia_losses)[0]

    if args['gamma'] == -1:
        for gamma in np.arange(0.5, 1.35, 0.05):
            pred[f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, ref_loss, rmia_losses)
    else:
        gamma = args['gamma']
        pred[f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, ref_loss, rmia_losses)
    return pred

def compute_decision_online(text, target_loss, ref_loss, target_model, ref_model, tokenizer1, tokenizer2, search_model, search_tokenizer, args):
    print(f"text = {text}")
    pred = {}

    # Loss
    pred["loss"] = target_loss.item()
    # Ratio of loss of large and small models
    pred["loss/ref_loss"] = (target_loss/ref_loss).item()
    
    if args['z_sampling'] == 'BERT_normalized':
        neighbors = generate_neighbors(text, args['num_z'], search_model, search_tokenizer)
    elif args['z_sampling'] == 'prefix':
        generate_args = [{'do_sample': True}]
        # generate_args = [{'do_sample': True, 'top_p': 0.9, 'temperature': 1.2}, {'do_sample': True, 'top_p': 0.8, 'temperature': 1.2}, {'do_sample': True, 'top_p': 0.7, 'temperature': 1.2}]
        # generate_args = [{'num_beams': args['num_z'], 'num_beam_groups': args['num_z'], 'diversity_penalty': 1.0}]
        for arg in generate_args:
            cur_args = copy.deepcopy(args)
            cur_args['generate_args'] = arg
            if args['prefix_length'] == -1:
                prefix_lens = [0.1, 0.15, 0.2]
            else:
                prefix_lens = [args['prefix_length']]
            for length in prefix_lens:
                cur_args['prefix_length'] = length
                res = evaluate_RMIA(text, target_loss, ref_loss, target_model, ref_model, tokenizer1, tokenizer2, cur_args)
                pred.update(res)
    else:
        print("ERROR: No z_sampling method specified for online MIA attack")

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

def jsonl_to_list(test_data):
    test_json = json.dumps(test_data)
    test_df = pd.read_json(io.StringIO(test_json))['input']
    test_list = test_df.astype(str).tolist()
    return test_list

def mia(test_data, rmia_data, target_model, ref_model, tokenizer1, tokenizer2, args):
    if rmia_data:
        print("Running offline RMIA")
        rmia_dl = DataLoader(rmia_data, batch_size=32, shuffle=False)
        target_losses_z = evaluate_model(target_model, tokenizer1, rmia_dl)
        ref_losses_z = evaluate_model(ref_model, tokenizer2, rmia_dl)
    else:
        print("Running online RMIA")
        CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
        search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
        search_model = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR).to('cuda')

    test_list = jsonl_to_list(test_data)
    test_dl = DataLoader(test_list, batch_size=32, shuffle=False)
    target_losses = evaluate_model(target_model, tokenizer1, test_dl)
    ref_losses = evaluate_model(ref_model, tokenizer2, test_dl)

    all_output = []
    for i in tqdm(range(len(test_data))): # [:100]
        if rmia_data:
            pred = compute_decision_offline(target_losses[i], ref_losses[i], target_losses_z, ref_losses_z, args)
        else:
            pred = compute_decision_online(test_list[i], target_losses[i], ref_losses[i], target_model, ref_model, tokenizer1, tokenizer2, search_model, search_tokenizer, args)
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
    parser.add_argument('-online', action='store_true')
    parser.add_argument('--input_path', type=str, default='data/newsSpace_oracle_tiny.jsonl')
    parser.add_argument('--z_sampling', type=str)
    parser.add_argument('--prefix_length', type=float, default=0.2)
    parser.add_argument('--description', type=str)
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

    # test data
    test_data = load_jsonl(args['input_path'])

    # RMIA data
    if not args['online']:
        rmia_path = 'data/newsSpace_oracle_val.csv'
        args['rmia_path'] = rmia_path
        rmia_data = pd.read_csv(rmia_path, names=['id', 'category', 'text'], header=0,  encoding='utf-8')['text'][:args['num_z']]
        rmia_data = rmia_data.astype(str).tolist()
    else:
        rmia_data = None

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
