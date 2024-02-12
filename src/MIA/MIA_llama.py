import logging
from os import replace
from weakref import ref
from sympy import true
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
from transformers import BertForMaskedLM, BertTokenizer, LogitsProcessorList
from datasets import load_dataset
from scipy.stats import norm
import math
import gc

from plots import fig_fpr_tpr
from load_utils import load_jsonl, load_model, read_jsonl, dump_jsonl, create_new_dir, jsonl_to_list
from neighbor_sampling import generate_neighbors, sample_generation, ContrastiveDecodingLogitsProcessor, PlausabilityLogitsProcessor
from utils import evaluate_model, get_idx_unreduced_loss, unzip_collate, ul

def RMIA_score(gamma, target_loss, ref_loss, rmia_losses):
    r"""
    Returns the classification score of an input for the MIA task, using the RMIA method (lower score corresponds to predicting an input was in the trainig dataset of the target model). For each neighbor z of x, we say x dominates z if target_loss(x) / ref_loss(x) * ref_loss(z) / target_loss(z) < gamma. The RMIA score is the fraction of z that x does not dominate. 
    
    gamma (float): The gamma used in the RMIA formula for thresholding dominance. 
    target_loss (torch.FloatTensor): The loss of the target model on x 
    ref_loss (torch.FloatTensor): The loss of the reference model on x
    rmia_losses (torch.FloatTensor): A sorted tensor of length equal to the number of neighbors z. Each entry contains the reference model's loss / target model's loss on a given z.

    Returns float of the RMIA score on x.
    """
    # Move terms of RMIA calculation to right side, then binary search the rmia_losses.
    threshold = gamma / target_loss * ref_loss 
    num_z_dominated = torch.searchsorted(rmia_losses, threshold).item()
    return 1 - num_z_dominated / len(rmia_losses)

def RMIA_score_original(gamma, target_loss, ref_loss, rmia_losses, replaced_indices):     
    num_z_dominated = 0 
    for i in range(len(replaced_indices)):
        unreduced_target_loss_original = torch.clone(target_loss)
        unreduced_ref_loss_original = torch.clone(ref_loss)
        for idx in replaced_indices[i]:
            unreduced_target_loss_original[idx] = 0
            unreduced_ref_loss_original[idx] = 0
        num_tokens_original = torch.sum(unreduced_target_loss_original != 0)

        target_loss_original = torch.sum(unreduced_target_loss_original) / num_tokens_original
        ref_loss_original = torch.sum(unreduced_ref_loss_original) / num_tokens_original
        
        if target_loss_original / ref_loss_original * rmia_losses[i] < gamma:
            num_z_dominated += 1
    return 1 - num_z_dominated / len(rmia_losses)

def RMIA_score_replaced(gamma, target_loss_tokens_x, ref_loss_tokens_x, target_losses_tokens_z, ref_losses_tokens_z, replaced_indices, args):     
    num_z_dominated = 0 
    num_z_dominated_no_perturb = 0
    for i in range(len(replaced_indices)):
        # unreduced_target_loss_replaced = torch.clone(target_loss)
        # unreduced_ref_loss_replaced = torch.clone(ref_loss)
        # for idx in range(len(unreduced_target_loss_replaced)):
        #     if idx not in set(replaced_indices[i]):
        #         unreduced_target_loss_replaced[idx] = 0
        #         unreduced_ref_loss_replaced[idx] = 0
        # num_tokens_replaced = torch.sum(unreduced_target_loss_replaced != 0)

        # target_loss_replaced = torch.sum(unreduced_target_loss_replaced) / num_tokens_replaced
        # ref_loss_replaced = torch.sum(unreduced_ref_loss_replaced) / num_tokens_replaced

        if args['last_idx_only']:
            loss_indices = sorted(replaced_indices[i])[-1:]
        else:
            loss_indices = replaced_indices[i]
        target_x = torch.sum(target_loss_tokens_x[loss_indices])/len(loss_indices)
        ref_x = torch.sum(ref_loss_tokens_x[loss_indices])/len(loss_indices)
        
        target_z = torch.sum(ul(target_losses_tokens_z, i)[loss_indices])/len(loss_indices)
        ref_z = torch.sum(ul(ref_losses_tokens_z, i)[loss_indices])/len(loss_indices)
        
        if target_x / ref_x < gamma:
            num_z_dominated_no_perturb += 1
        if target_x / ref_x * ref_z / target_z < gamma:
            num_z_dominated += 1
    # print(num_z_dominated_no_perturb, num_z_dominated)
    if args['no_perturbation']:
        num_z_dominated = num_z_dominated_no_perturb
    return 1 - num_z_dominated / len(replaced_indices)


def RMIA_score_squared(gamma, target_loss, ref_loss, rmia_losses):
    r"""
    RMIA with first term (target_loss_x / ref_loss_x) squared. 
    """
    # Move terms of RMIA calculation to right side, then binary search the rmia_losses.
    threshold = gamma * torch.square(ref_loss / target_loss)
    num_z_dominated = torch.searchsorted(rmia_losses, threshold).item()
    return 1 - num_z_dominated / len(rmia_losses)

def RMIA_score_sqrt(gamma, target_loss, ref_loss, rmia_losses):
    r"""
    RMIA with first term (target_loss_x / ref_loss_x) squared. 
    """
    # Move terms of RMIA calculation to right side, then binary search the rmia_losses.
    threshold = gamma * torch.sqrt(ref_loss / target_loss)
    num_z_dominated = torch.searchsorted(rmia_losses, threshold).item()
    return 1 - num_z_dominated / len(rmia_losses)

def evaluate_RMIA(text, target_loss, ref_loss, target_variance, ref_variance, unreduced_loss, unreduced_loss_ref, target_model, ref_model, generation_model, target_tokenizer, ref_tokenizer, generation_tokenizer, args):
    r"""
    Calculate the RMIA score for an input x, as well as related metrics. 

    Returns a dictionary mapping metric names to their values.

    Modify this function to try different RMIA attacks.
    """
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

    if args['idx_frac'] < 0:
        # Use multiple prefix lengths for neighbor generation.
        prefix_lens = [0.1, 0.15, 0.2]
    else:
        prefix_lens = [args['idx_frac']]

    res = {}
    # Name for all metrics begins with configuration arguments
    name = f"{args['generate_args']},prefix_len={prefix_lens} "

    neighbors = []
    replaced_indices = []
    if generation_model: 
        model_z = generation_model
        if 'logits_processor' in args['generate_args']:
            raise Exception("Contrastive decoding can't be used with a neutral generation model")
    elif args['ref_z']:
        # Neighbors are generated from reference model
        # If using contrastive decoding, the amateur model is the target model
        model_z = ref_model
        model_amateur = target_model
    else:
        # Neighbors are generated from target model
        # If using contrastive decoding, the amateur model is the reference model
        model_z = target_model
        model_amateur = ref_model
    
    # Generate neighbors
    if args['adaptive_idx']:
        indices = torch.argwhere(unreduced_loss - unreduced_loss_ref < -5)
        if len(indices) > 0: 
            idx = indices[-1]
            tokens = target_tokenizer(text).input_ids
            tokens = tokens[:idx+2]
            adaptive_prefix_len = len(target_tokenizer.decode(tokens))
        else:
            adaptive_prefix_len = 0
    else:
        adaptive_prefix_len = 0
    # print(unreduced_loss)
    # print(idx)
    # print(adaptive_prefix_len)

    for prefix_len in prefix_lens:
        # Generate neighbors using each prefix length
        # Configure the arguments for generation
        cur_args = copy.deepcopy(args)
        cur_args['idx_frac'] = prefix_len
        cur_args['adaptive_prefix_length'] = adaptive_prefix_len
        cur_args['num_z'] = args['num_z'] // len(prefix_lens)
        # Need to create a new logit processor for each generation
        if 'logits_processor' in args['generate_args']:
            # Use contrastive decoding
            # Must pop off dictionary, since cur_args['generate_args'] is directly passed into generate function
            contrastive_dec_alpha = cur_args['generate_args'].pop('contrastive_dec_alpha')
            min_probability_ratio = cur_args['generate_args'].pop('min_prob_ratio')
            cur_args['generate_args']['logits_processor'] = LogitsProcessorList([
                PlausabilityLogitsProcessor(min_probability_ratio),
                ContrastiveDecodingLogitsProcessor(model_amateur, contrastive_dec_alpha)
            ])
        # if cur_args['z_sampling'] == 'perturb':
        #     cur_neighbors, replaced_indices = sample_generation(text, model_z, target_tokenizer, cur_args)
        # else:
        #     cur_neighbors = sample_generation(text, model_z, target_tokenizer, cur_args)
        if cur_args['z_sampling'] == 'perturb':
            if cur_args['top_k']:
                cur_args['all_singles'] = True
                cur_args['z_per_indices'] = 10
            cur_neighbors, cur_replaced_indices = sample_generation(text, model_z, target_tokenizer, cur_args)
            neighbors.extend(cur_neighbors)
            replaced_indices.extend(cur_replaced_indices)
        else:
            cur_neighbors = sample_generation(text, model_z, target_tokenizer, cur_args)
            neighbors.extend(cur_neighbors)

    # if cur_args['top_k']:
    #     neighbors_dl = DataLoader(list(zip(neighbors, replaced_indices)), batch_size=32, shuffle=False, collate_fn=unzip_collate)
    #     target_losses_z, target_variances_z, target_losses_tokens_z, target_losses_z_original, target_losses_z_replaced = evaluate_model(target_model, target_tokenizer, neighbors_dl, replace=True)
    #     ref_losses_z, ref_variances_z, ref_losses_tokens_z, ref_losses_z_original, ref_losses_z_replaced = evaluate_model(ref_model, ref_tokenizer, neighbors_dl, replace=True)
        
    # 

    #     rmia_scores = []
    #     for i in range(RMIA_score_replaced(1.0, unreduced_loss, unreduced_loss_ref, target_losses_tokens_z, ref_losses_tokens_z, replaced_indices, cur_args)
        

    #     top_k_neighbors = []
    #     cur_args.pop('all_singles')
    #     cur_args['z_per_indices'] = args['z_per_indices']
    #     for prefix_len in prefix_lens:
    #         cur_neighbors, cur_replaced_indices = sample_generation(text, model_z, target_tokenizer, cur_args)
    #         neighbors.extend(cur_neighbors)
    #         replaced_indices.extend(cur_replaced_indices)
    
    if cur_args['z_sampling'] == 'perturb':
        neighbors_dl = DataLoader(list(zip(neighbors, replaced_indices)), batch_size=32, shuffle=False, collate_fn=unzip_collate)
    else:
        neighbors_dl = DataLoader(neighbors, batch_size=32, shuffle=False)
    target_losses_z, target_variances_z, target_losses_tokens_z, target_losses_z_original, target_losses_z_replaced = evaluate_model(target_model, target_tokenizer, neighbors_dl, replace=True)
    ref_losses_z, ref_variances_z, ref_losses_tokens_z, ref_losses_z_original, ref_losses_z_replaced = evaluate_model(ref_model, ref_tokenizer, neighbors_dl, replace=True)
    

    # Count fraction of z that x has higher loss than for the target model
    ratio = torch.count_nonzero(target_losses_z < target_loss).item() / len(target_losses_z)
    res[name + f'neighbor'] = ratio

    # Count fraction of z that x has higher loss than for the reference model
    ratio = torch.count_nonzero(ref_losses_z < ref_loss).item() / len(ref_losses_z)
    res[name + f'ref_neighbor'] = ratio

    # Neighborhood attack, using mean on neighbors
    res[name + 'loss/neighbors'] = (target_loss / torch.mean(target_losses_z)).item()
    if math.isnan(res[name + 'loss/neighbors']):
        # Error
        print(target_losses_z)
        print(neighbors)

    res[name + 'mean(loss/neighbors)'] =torch.mean(target_loss / target_losses_z).item()
    
    # Neighborhood attack, using cdf on neighbors
    # res[name + 'loss neighbor gaussians'] = norm.cdf(target_loss.item(), loc=torch.mean(target_losses_z).item(), scale=torch.std(target_losses_z).item())

    # RMIA attack
    rmia_losses = ref_losses_z / target_losses_z 
    rmia_losses = torch.sort(rmia_losses)[0]
    rmia_vars = ref_variances_z / target_variances_z
    rmia_vars = torch.sort(rmia_vars)[0]

    if args['gamma'] == -1:
        for gamma in np.arange(0.5, 1.5, 0.05):
            # Try different gammas
            # RMIA attack using loss 
            gamma = np.round(gamma, 2)
            if not args['no_perturbation']:
                res[name + f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, ref_loss, rmia_losses)
            # res[name + f"RMIA^2(gamma={gamma})"] = RMIA_score_squared(gamma, target_loss, ref_loss, rmia_losses)
            # res[name + f"RMIA^1/2(gamma={gamma})"] = RMIA_score_sqrt(gamma, target_loss, ref_loss, rmia_losses)

            if cur_args['z_sampling'] == 'perturb':
                if not args['no_perturbation']:
                    res[name + f"RMIA_original(gamma={gamma})"] = RMIA_score_original(gamma, unreduced_loss, unreduced_loss_ref, ref_losses_z_original /target_losses_z_original, replaced_indices)
                res[name + f"RMIA_replaced(gamma={gamma})"] = RMIA_score_replaced(gamma, unreduced_loss, unreduced_loss_ref, target_losses_tokens_z, ref_losses_tokens_z, replaced_indices, cur_args)
            # RMIA attack using variance
            # res[name + f"RMIA_var(gamma={gamma})"] = RMIA_score(gamma, target_variance, ref_variance, rmia_vars)
    else:
        gamma = args['gamma']
        res[name + f"RMIA(gamma={gamma})"] = RMIA_score(gamma, target_loss, ref_loss, rmia_losses)
        res[name + f"RMIA_var(gamma={gamma})"] = RMIA_score(gamma, target_variance, ref_variance, rmia_vars)

    # RMIA attack where the mean of the neighbors is used instead of individual comparisons.
    res[name + 'RMIA mean'] = (target_loss / ref_loss * torch.mean(rmia_losses)).item()

    res[name + 'RMIA mean ratios'] = torch.mean(target_loss / ref_loss * rmia_losses).item()
    return res

def compute_decision_offline(target_loss, ref_loss, target_losses_z, ref_losses_z, args):
    # Offline MIA attacks that do not do additional work on each input x. 
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

def compute_decision_online(text, target_loss, ref_loss, target_variance, ref_variance, unreduced_loss, unreduced_loss_ref, target_model, ref_model, generation_model, target_tokenizer, ref_tokenizer, generation_tokenizer, search_model, search_tokenizer, args):
    # Do all the MIA attacks, and use the online version of RMIA 
    torch.cuda.empty_cache()
    print(f"text = {text}")
    pred = {}

    # zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

    # Loss attack
    pred["loss"] = target_loss.item()
    # Loss attack with entropy 
    # pred["loss/zlib"] = (target_loss / zlib_entropy).item()
    # Reference attack
    pred["loss/ref_loss"] = (target_loss/ref_loss).item()
    # Reference attack with entropy
    # pred["loss/ref_loss / zlib"] = (target_loss/(ref_loss * zlib_entropy)).item()
    # Variance attack
    # pred["variance"] = target_variance.item()
    # # Variance attack with entropy
    # pred["variance/zlib"] = (target_variance / zlib_entropy).item()
    # # Reference variance attack
    # pred["variance/ref_variance"] = (target_variance / ref_variance).item()
    # # Reference variance attack with entropy
    # pred["variance/ref_variance / zlib"] = (target_variance / (ref_variance * zlib_entropy)).item()

    if args['z_sampling'] == 'BERT_normalized':
        # Sample z with BERT
        neighbors = generate_neighbors(text, args['num_z'], search_model, search_tokenizer)
    elif args['z_sampling'] == 'prefix' or args['z_sampling'] == 'perturb':
        # Sample z by generating from prefix
        generate_args = [{'do_sample': True}]
        if args["contrastive_dec"]:
            for contrastive_dec_alpha in [1]:
                generate_args.append({'do_sample': True, 'logits_processor': True, 'contrastive_dec_alpha': contrastive_dec_alpha, 'min_prob_ratio': 0.01})
        for arg in generate_args:
            cur_args = copy.deepcopy(args)
            cur_args['generate_args'] = arg
            res = evaluate_RMIA(text, target_loss, ref_loss, target_variance, ref_variance, unreduced_loss, unreduced_loss_ref, target_model, ref_model, generation_model, target_tokenizer, ref_tokenizer, generation_tokenizer, cur_args)
            pred.update(res)
    elif args['z_sampling'] == 'select_indices':
        res = evaluate_RMIA_select(text, unreduced_loss, unreduced_loss_ref, target_tokenizer, args)
        pred.update(res)
    # else:
    #     print("ERROR: No z_sampling method specified for online MIA attack")
    
    return pred

def evaluate_RMIA_select(text, unreduced_loss, unreduced_loss_ref, target_tokenizer, args):
    res = {}
    num_tokens = len(target_tokenizer(text)['input_ids'])-1
    loss_ratios = unreduced_loss[:num_tokens] / unreduced_loss_ref[:num_tokens]
    if args['max_indices'] == 1:
        if args['gamma'] == -1:
            for gamma in np.arange(0.75, 1.25, 0.05):
                # Try different gammas
                gamma = np.round(gamma, 2)
                res[f"RMIA_replaced(gamma={gamma})"] = 1-(torch.sum(loss_ratios < gamma)/num_tokens).item()
        else:
            gamma = args['gamma']
            res[f"RMIA_replaced(gamma={gamma})"] = 1-(torch.sum(loss_ratios < gamma)/num_tokens).item()
    return res

def mia(test_data, rmia_data, target_model, ref_model, generation_model,target_tokenizer, ref_tokenizer, generation_tokenizer, args):
    if rmia_data:
        print("Running offline RMIA")
        rmia_dl = DataLoader(rmia_data, batch_size=32, shuffle=False)
        target_losses_z, target_variances_z, _, _, _ = evaluate_model(target_model, target_tokenizer, rmia_dl)
        ref_losses_z, ref_variances_z, _, _, _ = evaluate_model(ref_model, ref_tokenizer, rmia_dl)
    # else:
    #     print("Running online RMIA")
    #     CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
    #     search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
    #     search_model = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR).to('cuda')

    test_list = jsonl_to_list(test_data)
    test_dl = DataLoader(test_list, batch_size=32, shuffle=False)
    target_losses, target_variances, unreduced_losses, _, _ = evaluate_model(target_model, target_tokenizer, test_dl) 
    ref_losses, ref_variances, unreduced_losses_ref, _, _ = evaluate_model(ref_model, ref_tokenizer, test_dl)

    all_output = []
    for i in tqdm(range(len(test_data))): # [:100]
        if i % 50 == 0:
            gc.collect()
        if rmia_data:
            pred = compute_decision_offline(target_losses[i], ref_losses[i], target_losses_z, ref_losses_z, args)
        else:
            pred = compute_decision_online(test_list[i], target_losses[i], ref_losses[i], target_variances[i], ref_variances[i], 
            get_idx_unreduced_loss(unreduced_losses, i), 
            get_idx_unreduced_loss(unreduced_losses_ref, i), target_model, ref_model, generation_model, target_tokenizer, ref_tokenizer, generation_tokenizer, None, None, args)
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
    parser.add_argument('--gamma', type=float, default=1, help="Gamma for RMIA. Set to -1 to use multiple gamma values.")
    parser.add_argument('--target_path', type=str, help="Checkpoint path for the target model")
    parser.add_argument('--target_model', type=str, default='gpt2', help="Model name of the target model, gpt2 by default")
    parser.add_argument('--ref_path', type=str, default=None, help="Checkpoint path for the reference model (optional)")
    parser.add_argument('--ref_model', type=str, default='gpt2', help="Model name of the reference model, gpt2 by default")
    parser.add_argument('--generation_model', type=str, default=None, help="Model to generate neighbors")
    parser.add_argument('--generation_path', type=str, default=None, help="Checkpoint path for the generation model (optional)")
    parser.add_argument('--num_z', type=int, default = 100, help="Number of neighbors to generate for each input")
    parser.add_argument('--z_per_indices', type=int, default = 1, help="Number of neighbors to generate for each set of perturbed indices")
    parser.add_argument('-offline', action='store_true', help="Use offline RMIA")
    parser.add_argument('--input_path', type=str, default='data/newsSpace_oracle_tiny.jsonl', help="Path to the dataset")
    parser.add_argument('--z_sampling', type=str, help="Method to sample neighbors. Use 'prefix' to generate neighbors from a prefix of the input. Use 'perturb' to generate neighbors through perturbing input.")
    parser.add_argument('-ref_z', action='store_true', help="Generate neighbors from the reference model instead of the target model")
    parser.add_argument('-contrastive_dec', action='store_true', help="Use contrastive decoding when generating neighbors")
    parser.add_argument('-adaptive_idx', action='store_true', help="Adaptively determine prefix len to include rightmost token where target model significantly outperforms ref model")
    parser.add_argument('--description', type=str, help="Notes to add to this run (all the arguments get dumped into the output)")
    parser.add_argument('--idx_frac', type=float, help="Maximum fraction of indices to perturb")
    parser.add_argument('--max_indices', type=int, default=0, help="Maximum number of indices to perturb")
    parser.add_argument('-perturb_span', action='store_true', help="Perturb spans instead of individual indices")
    parser.add_argument('-fixed', action='store_true', help="Perturb a fixed fraction of indices")
    parser.add_argument('-perturb_generation', action='store_true', help="Generate perturbations from model")
    parser.add_argument('-last_idx_only', action='store_true', help="Compute RMIA scores only using the last (rightmost) replaced indices")
    parser.add_argument('-no_perturbation', action='store_true', help="Don't actually perturb the sentence. Used as ablation for index selection.")
    parser.add_argument('-tokenizer_perturbation', action='store_true', help="Perturb the input only through encoding and decoding with tokenizer")
    parser.add_argument('-top_k', action='store_true', help="Select the top k indices with lowest RMIA score to generate neighbors from.")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = get_cli_args()

    target_model, target_tokenizer, = load_model(args['target_model'], path=args['target_path'])
    ref_model, ref_tokenizer, = load_model(args['ref_model'], path=args['ref_path'])
    if args['generation_model']:
        generation_model, generation_tokenizer, = load_model(args['generation_model'], path=args['generation_path'])
    else:
        generation_model = generation_tokenizer = None
    
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
    if args['offline']:
        rmia_path = 'data/newsSpace_oracle_val.csv'
        args['rmia_path'] = rmia_path
        rmia_data = pd.read_csv(rmia_path, names=['id', 'category', 'text'], header=0,  encoding='utf-8')['text'][:args['num_z']]
        rmia_data = rmia_data.astype(str).tolist()
    else:
        rmia_data = None

    with open(output_dir + '/args.json', mode='w') as f:
        json.dump(args, f)

    all_output = mia(test_data, rmia_data, target_model, ref_model, generation_model, target_tokenizer, ref_tokenizer, generation_tokenizer, args)
    '''
    dump and read data 
    '''
    dump_jsonl(all_output, f"{output_dir}/output.jsonl")
    all_output = read_jsonl(f"{output_dir}/output.jsonl")
    
    '''
    plot
    '''
    fig_fpr_tpr(all_output, output_dir)
    print("===============================")
