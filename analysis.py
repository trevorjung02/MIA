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

CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=CACHE_DIR)
# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
target_model.config.pad_token_id = target_model.config.eos_token_id
target_model.generation_config.pad_token_id = target_model.config.eos_token_id
path = "/mmfs1/gscratch/ark/tjung2/MIA/checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533"
ckpt = torch.load(path)
target_model.load_state_dict(ckpt['model_state_dict'])
target_model.eval()
target_model.cuda()

ref_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=CACHE_DIR)
# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
ref_model.config.pad_token_id = ref_model.config.eos_token_id
ref_model.generation_config.pad_token_id = ref_model.config.eos_token_id
ref_model.eval()
ref_model.cuda()

path = '/mmfs1/gscratch/ark/tjung2/MIA/output/agnews/epoch-3_perplexity-20.9533/none/run_200/output.jsonl'
with open(path) as f:
    df = pd.read_json(f, lines=True)

def evaluate_model(model, tokenizer, dl, replace=False):
    with torch.no_grad():
        losses = []
        losses_original = []
        losses_replaced = []
        variances = []
        unreduced_losses = []
        for sentence_batch in dl:
            # print(sentence_batch)
            if replace:
                sentences, replaced_indices = sentence_batch
            else:
                sentences = sentence_batch
            batch = tokenizer(sentences, padding=True, return_tensors='pt', truncation=True, max_length=150)
            labels = torch.tensor([
                [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch['attention_mask'], batch['input_ids'])]
                ])
            batch['labels'] = labels
            batch = batch.to(model.device)
            
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            unreduced_loss = loss_fct(shift_logits.transpose(1,2), shift_labels)

            unreduced_losses.append(unreduced_loss)
            num_tokens = torch.sum(shift_labels != -100, dim=1)

            for i in range(unreduced_loss.size(dim=0)):
                variances.append(torch.var(unreduced_loss[i][:num_tokens[i]], dim=0, keepdim=True))
            
            loss_sum = torch.sum(unreduced_loss, dim=1)
            loss = loss_sum / num_tokens
            # print(loss)
            losses.append(loss)
            
            if replace:
                num_replaced_indices = torch.tensor([len(row) for row in replaced_indices], device=model.device)
                num_original_indices = num_tokens - num_replaced_indices

                unreduced_loss_original = torch.clone(unreduced_loss)
                for i in range(len(replaced_indices)):
                    for idx in replaced_indices[i]:
                        unreduced_loss_original[i][idx] = 0
                loss_sum_original = torch.sum(unreduced_loss_original, dim=1)
                losses_original.append(loss_sum_original / num_original_indices)

                unreduced_loss_replaced = torch.clone(unreduced_loss)
                for i in range(len(replaced_indices)):
                    for idx in range(len(unreduced_loss_replaced[i])):
                        if idx not in set(replaced_indices[i]):
                            unreduced_loss_replaced[i][idx] = 0
                loss_sum_replaced = torch.sum(unreduced_loss_replaced, dim=1)
                losses_replaced.append(loss_sum_replaced / num_replaced_indices)

        variances = torch.cat(variances)
        losses = torch.cat(losses)
        if replace:
            losses_original = torch.cat(losses_original)
            losses_replaced = torch.cat(losses_replaced)
        # with torch.no_grad():
        #     outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        #     losses.append(outputs.loss)
    return losses, variances, unreduced_losses, losses_original, losses_replaced

def unzip_collate(batch):
        return list(zip(*batch))

def get_indices_individual(idx_frac, num_tokens, fixed):
    if fixed:
        num_replacements = max(math.ceil(idx_frac * num_tokens), 1)
    else:
        num_replacements = max(np.random.randint(0, math.ceil(idx_frac * num_tokens)), 1)
    # num_replacements = max(int(0.2 * num_tokens+1), 1)
    indices = np.random.choice(num_tokens, num_replacements, replace=False)
    return indices
    
def sample_generation(sentence, model, tokenizer, args):
    if args['z_sampling'] == 'perturb':
        # print(f"Generating samples from perturbation, with idx_frac = {args['idx_frac']}")
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
            input_ids = input_ids.to(model.device)
            logits = model(input_ids).logits[0][:-1]

            # for i in range(len(logits)):
            #     logits[i][input_ids[0][i+1]] = -float("Inf")
                
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # for i in range(len(probs)):
            #     probs[i][input_ids[0][i+1]] = 0

            neighbors = []
            replaced_indices = []
            new_tokens = []
            for _ in range(args['num_z']):
                if args['indices'] is not None:
                    indices = args['indices']
                else:
                    indices = get_indices_individual(args['idx_frac'], len(probs), args['fixed'])
                neighbor = input_ids[0].clone()
                cur_replaced_indices = []
                cur_new_tokens = []
                for idx in indices:
                    tokens = torch.multinomial(probs[idx], 1, replacement=True)
                    if neighbor[idx+1] != tokens[0]:
                        neighbor[idx+1] = tokens[0]
                        cur_replaced_indices.append(idx)
                        cur_new_tokens.append(tokens[0])
                neighbors.append(neighbor)
                replaced_indices.append(cur_replaced_indices)
                new_tokens.append(cur_new_tokens)
            complete_generated_text = tokenizer.batch_decode(neighbors, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return complete_generated_text, replaced_indices, new_tokens
    else:
        print(f"Generating samples from {model.name_or_path}")
        half_sentence_index = math.ceil(len(sentence.split())*args['idx_frac'])
        if args['adaptive_prefix_length'] < len(sentence.split())-1 and args['adaptive_prefix_length'] > half_sentence_index:
            half_sentence_index = args['adaptive_prefix_length']
        # half_sentence_index = 8
        if half_sentence_index > 0:
            prefix = " ".join(sentence.split()[:half_sentence_index])
        else:
            prefix = '<|startoftext|> '
        # continuation = " ".join(sentence.split()[half_sentence_index:])
        print(f"Generating from prefix {prefix}")
        # print(continuation)
        
        input_ids = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        # print(input_ids)

        output = model.generate(input_ids, max_new_tokens=len(sentence.split())-half_sentence_index, min_new_tokens=1, num_return_sequences=args['num_z'], pad_token_id=tokenizer.eos_token_id, **args['generate_args'])
        # print(output)
        complete_generated_text = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(complete_generated_text)
        # generated_text = complete_generated_text[:, len(prefix):]
        # generated_text = tokenizer.batch_decode(output[:, len(input_ids[0]):], skip_special_tokens=True)
        # print(generated_text)
        return complete_generated_text
    
idx = 2855
sentence = df['input'].iloc[idx]
test_dl = DataLoader([sentence], batch_size=32, shuffle=False)
target_losses, target_variances, unreduced_losses, _, _ = evaluate_model(target_model, tokenizer, test_dl) 
ref_losses, ref_variances, unreduced_losses_ref, _, _ = evaluate_model(ref_model, tokenizer, test_dl)

def get_neighbors(indices=None):
    args = {
        'z_sampling': 'perturb',
        'num_z': 250,
        'idx_frac': 0.1,
        'fixed': True,
        'perturb_span': False,
        'perturb_generation': False,
        'indices': indices
    }
    neighbors, replaced_indices, new_tokens = sample_generation(sentence, target_model, tokenizer, args)
    return neighbors, replaced_indices

def ul(unreduced_losses, idx, batch_size=32):
    return unreduced_losses[idx //  batch_size][idx % batch_size]

def compute_rmia_score(neighbors, replaced_indices, last_idx_only=False):
    rmia_ratios = []
    num_z_dominated = 0

    neighbor_dl = DataLoader(neighbors, batch_size=32, shuffle=False)
    unreduced_losses_z = evaluate_model(target_model, tokenizer, neighbor_dl)[2]
    unreduced_losses_ref_z = evaluate_model(ref_model, tokenizer, neighbor_dl)[2]
    
    for idx in range(len(replaced_indices)):
        # print(replaced_indices[idx])
        if last_idx_only:
            loss_indices = sorted(replaced_indices[idx])[-1:]
        else:
            loss_indices = replaced_indices[idx]
        target_z = torch.sum(ul(unreduced_losses_z, idx)[loss_indices])/len(loss_indices)
        target_x = torch.sum(ul(unreduced_losses, 0)[loss_indices])/len(loss_indices)
    
        ref_z = torch.sum(ul(unreduced_losses_ref_z, idx)[loss_indices])/len(loss_indices)
        ref_x = torch.sum(ul(unreduced_losses_ref, 0)[loss_indices])/len(loss_indices)
    
        ratio = target_x / ref_x * ref_z / target_z
        # print("({0:.2f}/{1:.2f}) / ({3:.2f}/{2:.2f}) = {4:.2f}".format(target_x, ref_x, ref_z, target_z, ratio.item()))
        if ratio < 1:
            num_z_dominated += 1
        
        if not torch.isnan(ratio):
            rmia_ratios.append((ratio.item(), target_x, ref_x, target_z, ref_z))
    # rmia_ratios.sort(key=lambda x: x[0])
    # mean = statistics.mean([x[0] for x in rmia_ratios])
    return 1-num_z_dominated/len(replaced_indices)

rmia_scores = []
for i in range(len(tokenizer(sentence)['input_ids']) - 1):
    indices = [i]
    neighbors, replaced_indices = get_neighbors(indices)
    rmia_score = compute_rmia_score(neighbors, replaced_indices)
    print(f"{indices}: {rmia_score}")
    rmia_scores.append(rmia_score)

def get_pairs(low, high):
    pairs = []
    for i in range(low, high-1):
        for j in range(i+1, high):
            pairs.append((i,j))
    return pairs

def create_new_dir(dir: str) -> str:
    if not os.path.exists(dir):
        num = 1
    else:
        files = set(os.listdir(dir))
        num = len(files)+1
        while f"run_{num}" in files:
            num += 1
    new_dir = os.path.join(dir, f"run_{num}")
    os.makedirs(new_dir, exist_ok=False)
    return new_dir

logger = logging.getLogger()
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])

dir = create_new_dir('analysis/pairs')
path = os.path.join(dir, 'rmia_scores.txt')
logging.basicConfig(filename=path, level=logging.DEBUG, format='')

num_tokens = len(tokenizer(sentence)['input_ids'])-1
pair_to_rmia = {}
for pair in get_pairs(0, num_tokens):
    logging.info(pair)
    indices = np.argsort(rmia_scores)[list(pair)]
    neighbors, replaced_indices = get_neighbors(indices)
    logging.info(replaced_indices[:20])
    rmia_score = compute_rmia_score(neighbors, replaced_indices, last_idx_only=False)
    pair_to_rmia[pair] = rmia_score
    logging.info(f"{indices}: {rmia_score}")

