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
    
def sample_generation(sentence, model, tokenizer, args):
    if args['perturb']:
        print(f"Generating samples from perturbation, with idx_frac = {args['idx_frac']}")
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
            input_ids = input_ids.to(model.device)
            logits = model(input_ids).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0][:-1]
            neighbors = []
            
            for _ in range(args['num_z']):
                neighbor = input_ids[0].clone()
                num_replacements = max(np.random.randint(0, int(args['idx_frac'] * len(probs)+1), 1), 1)
                # num_replacements = max(int(0.2 * len(probs)+1), 1)
                indices = np.random.choice(len(probs), num_replacements, replace=False)
                for idx in indices:
                    tokens = torch.multinomial(probs[idx], 1, replacement=True)
                    neighbor[idx+1] = tokens[0]
                neighbors.append(neighbor)
            complete_generated_text = tokenizer.batch_decode(neighbors, skip_special_tokens=True)
    else:
        print(f"Generating samples from {model.name_or_path}")
        half_sentence_index = math.ceil(len(sentence.split())*args['prefix_length'])
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
        complete_generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
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