import torch
from heapq import nlargest
import math

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
    half_sentence_index = math.ceil(len(sentence.split())*args['prefix_length'])
    # half_sentence_index = 8
    if half_sentence_index > 0:
        prefix = " ".join(sentence.split()[:half_sentence_index])
    else:
        prefix = '<|startoftext|> '
    # continuation = " ".join(sentence.split()[half_sentence_index:])
    # print(prefix)
    # print(continuation)
    
    input_ids = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    # print(input_ids)
    output = model.generate(input_ids, max_new_tokens=len(sentence.split())-half_sentence_index, num_return_sequences=args['num_z'], pad_token_id=tokenizer.eos_token_id, **args['generate_args'])
    # print(output)
    complete_generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print(complete_generated_text)
    # generated_text = complete_generated_text[:, len(prefix):]
    # generated_text = tokenizer.batch_decode(output[:, len(input_ids[0]):], skip_special_tokens=True)
    # print(generated_text)
    return complete_generated_text