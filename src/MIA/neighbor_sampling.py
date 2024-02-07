import token
import torch
from heapq import nlargest
import math
from transformers import LogitsWarper, LogitsProcessorList, LogitsProcessor
import copy
import numpy as np

def get_alt_text(original_text, decoded_alt_text):
    lower_text = original_text.lower()
    alt_split = decoded_alt_text.split()
    i = 0
    for word_index in range(len(alt_split)):
        w = alt_split[word_index]
        while i < len(lower_text) and lower_text[i].isspace():
            i += 1
        j = i
        for c in w:
            if j < len(lower_text) and lower_text[j] == c:
                j += 1
            else:
                # print(w)
                if word_index+1 < len(alt_split):
                    next_word = alt_split[word_index+1]
                    remaining_text = lower_text[j:]
                    r = get_right_index(remaining_text, next_word)
                    if r < 0:
                        print(f"ERROR: cannot find {next_word} in {remaining_text}")
                        return None
                    alt_text = original_text[:i] + w + original_text[j+r:]
                else:
                    alt_text = original_text[:i] + w
                return alt_text
        i = j
    print(f"ERROR: alt_text is the same as original_text")
    return None

def get_right_index(s, w):
    i = s.find(w) - 1
    while i >= 0 and s[i].isspace():
        i -= 1
    return i+1

def generate_neighbors(text, max_neighbors, search_model, search_tokenizer):
    # print(f"text = {text}")
    # print("-------------------------------------")
    with torch.no_grad():
        text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 150, return_tensors='pt').input_ids.to('cuda')
    
        candidate_scores = dict()
        token_dropout = torch.nn.Dropout(p=0.7)
    
        for target_token_index in range(1, len(text_tokenized[0,:])-1):
        # for target_token_index in [1]:
            target_token = text_tokenized[0,target_token_index]
            embeds = search_model.bert.embeddings(text_tokenized)
            embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=1), embeds[:,target_token_index+1:,:]), dim=1)
            
            token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)
    
            original_prob = token_probs[0,target_token_index, target_token]
    
            top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 10, dim=1)
    
            for cand, prob in zip(top_candidates[0], top_probabilities[0]):
                if not cand == target_token:
                    alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda'), text_tokenized[:,target_token_index+1:]), dim=1)
                    decoded_alt_text = search_tokenizer.batch_decode(alt, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    # alt_text = search_tokenizer.batch_decode(alt.transpose(0,1))
                    # print(f"decoded_alt_text = {decoded_alt_text}")
                    
                    alt_text = get_alt_text(text, decoded_alt_text)
                    if not alt_text:
                        continue
                    # print(f"alt_text = {alt_text}")
                    # print("------------------------------------")
    
                    # cand_text = search_tokenizer.decode(cand)
                    # candidate_scores[(decoded_alt_text, alt_text, cand_text, prob/(1-original_prob))] = prob/(1-original_prob)
                    candidate_scores[alt_text] = prob/(1-original_prob)
    
        highest_scored_texts = nlargest(max_neighbors, candidate_scores, key = candidate_scores.get)
        return highest_scored_texts

def get_indices_span(idx_frac, num_tokens, fixed):
    if fixed:
        total_perturbations = max(math.ceil(idx_frac * num_tokens), 1)
    else:
        total_perturbations = max(np.random.randint(0, math.ceil(idx_frac * num_tokens), 1)[0], 1)
    indices = set()
    while len(indices) < total_perturbations:
        num_replacements = np.random.default_rng().geometric(p=0.5, size=1)[0]
        # print(num_replacements)
        num_replacements = np.maximum(num_replacements, 1)
        num_replacements = np.minimum(num_replacements, total_perturbations-len(indices))
        # print(num_replacements)
        start_index = np.random.choice(num_tokens, 1, replace=False)[0]
        # print(start_index)
        for idx in range(start_index, min(start_index+num_replacements, num_tokens)):
            indices.add(idx)
    return list(indices)

def get_indices_individual(idx_frac, num_tokens, fixed, max_indices=None, idx_set=None):
    if idx_set:
        num_tokens = len(idx_set)
    if fixed:
        num_replacements = max(math.ceil(idx_frac * num_tokens), 1)
    else:
        num_replacements = max(np.random.randint(0, math.ceil(idx_frac * num_tokens)), 1)
    if max_indices > 0:
        num_replacements = min(num_replacements, max_indices)
    # num_replacements = max(int(0.2 * num_tokens+1), 1)
    if idx_set:
        indices = np.random.choice(idx_set, num_replacements, replace=False)
    else:
        indices = np.random.choice(num_tokens, num_replacements, replace=False)
    return indices

def generate_perturbation(input_ids, model, indices):
    indices.sort()
    cache = None
    cur_replaced_indices = []
    with torch.no_grad():
        output_tokens = input_ids[0].clone()
        for i in range(len(indices)):
            # print(i)
            if indices[i] == 0:
                continue
            if i == 0:
                left = 0
            else:
                left = indices[i-1]+1
            # print(f"{left}:{indices[i]}")
            input = output_tokens[left:indices[i]+1]
            if cache:
                args = {
                    'past_key_values': cache
                }
            else:
                args = {}
            output = model(input.unsqueeze(dim=0), use_cache=True, return_dict=True, **args)
            cache = output.past_key_values
            probs = torch.nn.functional.softmax(output.logits[0][-1], dim=-1)
            token = torch.multinomial(probs, 1, replacement=True)
            
            if output_tokens[indices[i]+1] != token:
                output_tokens[indices[i]+1] = token
                cur_replaced_indices.append(indices[i])

            # output_tokens[indices[i]] = torch.multinomial(probs, 1, replacement=True)
        return output_tokens, cur_replaced_indices


def sample_generation(sentence, model, tokenizer, args):
    if args['z_sampling'] == 'perturb':
        print(f"Generating samples from perturbation, with idx_frac = {args['idx_frac']}")
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
            # print(f"Number of index sets: {args['num_z'] // args['z_per_indices']}")
            if 'all_singles' in args:
                num_index_sets = len(probs)
            else:
                num_index_sets = args['num_z'] // args['z_per_indices']
            for i in range(num_index_sets):
                if 'all_singles' in args:
                    indices = [i]
                elif args['perturb_span']:
                    indices = get_indices_span(args['idx_frac'], len(probs), args['fixed'])
                else:
                    if 'indices' in args:
                        num_possible_indices = len(args['indices'])
                        indices = get_indices_individual(args['idx_frac'], num_possible_indices, args['fixed'], args['max_indices'], args['indices'])
                    else:
                        num_possible_indices = min(len(probs), 149)
                        indices = get_indices_individual(args['idx_frac'], num_possible_indices, args['fixed'], args['max_indices'])
                
                for _ in range(args['z_per_indices']):
                    if args['perturb_generation']:
                        neighbor, cur_replaced_indices = generate_perturbation(input_ids, model, indices)
                    else:
                        neighbor = input_ids[0].clone()
                        cur_replaced_indices = []
                        for idx in indices:
                            tokens = torch.multinomial(probs[idx], 1, replacement=True)
                            if args['no_perturbation'] or args['tokenizer_perturbation']:
                                cur_replaced_indices.append(idx)
                            else:
                                if neighbor[idx+1] != tokens[0]:
                                    neighbor[idx+1] = tokens[0]
                                    cur_replaced_indices.append(idx)
                    neighbors.append(neighbor)
                    replaced_indices.append(cur_replaced_indices)
            # print(f"Number of neighbors: {len(neighbors)}")
            if args['no_perturbation']:
                complete_generated_text = [sentence] * args['num_z']
            else:
                complete_generated_text = tokenizer.batch_decode(neighbors, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print(tokenizer(complete_generated_text, return_length=True).length)
            return complete_generated_text, replaced_indices
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

class ContrastiveDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, amateur, alpha) -> None:
        self.amateur = amateur
        self.input_ids = torch.empty(1).to(amateur.device)
        self.past_key_values = None
        self.alpha = alpha

    def __call__(self, input_ids, scores):            
        # print(f"input_ids size = {input_ids.size()}")
        with torch.no_grad():
            if torch.equal(self.input_ids, input_ids[:,:-1]):
                amateur_out = self.amateur(input_ids[:,-1].unsqueeze(dim=1), return_dict=True, past_key_values = self.past_key_values, use_cache=True)
            else:
                amateur_out = self.amateur(input_ids, return_dict=True, use_cache=True)
        self.input_ids = input_ids
        self.past_key_values = amateur_out.past_key_values
        # print((self.past_key_values[0][0].size()))
        # print(amateur_out.logits.size())
        return scores - self.alpha * (amateur_out.logits[:,-1,:])
    
class PlausabilityLogitsProcessor(LogitsProcessor):
    def __init__(self, alpha: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        alpha = float(alpha)
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"`alpha` has to be a float > 0 and < 1, but is {alpha}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.alpha = torch.tensor(alpha)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        
    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        # Calculate the adaptive cutoff
        probabilities = scores.softmax(dim=-1)
        indices_to_remove = probabilities < self.alpha * torch.max(probabilities)

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores