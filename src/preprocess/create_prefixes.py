from argparse import ArgumentParser
import torch
from datasets import load_dataset
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def sample_generation(sentence, model, tokenizer, args):
    half_sentence_index = math.ceil(len(sentence.split())*args['idx_frac'])
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
    
def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--target_path', type=str, help="Checkpoint path for the target model")
    parser.add_argument('--target_model', type=str, default='gpt2', help="Model name of the target model, gpt2 by default")
    parser.add_argument('--num_z', type=int, default = 100, help="Number of neighbors to generate for each input")
    parser.add_argument('--idx_frac', type=float, help="Maximum fraction of indices to perturb")
    args = parser.parse_args()
    return vars(args)

def load_model(name, path=None):
    # CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
    CACHE_DIR = "/gscratch/zlab/swj0419/huggingface_models"
    
    if "pythia" in name:
        model = GPTNeoXForCausalLM.from_pretrained(name, return_dict=True, 
        cache_dir=CACHE_DIR).cuda()
    elif "gpt2" in name:
            model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, cache_dir=CACHE_DIR).cuda()
            model.config.pad_token_id = model.config.eos_token_id
            model.generation_config.pad_token_id = model.config.eos_token_id
    elif "llama" in name:
        CACHE_DIR = "/gscratch/zlab/swj0419/huggingface_models"
        model = AutoModelForCausalLM.from_pretrained(
           pretrained_model_name_or_path=name,
           torch_dtype=torch.bfloat16,
           trust_remote_code=True,
           low_cpu_mem_usage=True,
           device_map="auto",
           load_in_8bit=True,
            cache_dir=CACHE_DIR
        )
    
    if path:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    if name == "swj0419/7b_finetuned_llama2_3epoch":
        name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
    
def main():
    CACHE_DIR = '/gscratch/zlab/swj0419/MIA/huggingface'
    LENGTH = 128
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

    # args = get_cli_args()
    args = {
        "target_path": None,
        '--target_model': 'gpt2',
        '--num_z': 1,
        '--idx_frac': 0.5,
        'generate_args': {'do_sample': True}
    }
    target_model, target_tokenizer, = load_model(args['target_model'], path=args['target_path'])
    
    df = pd.DataFrame(dataset)
    train_df = df[df['label'] ==1 ]
    train_data = train_df['input'].tolist()
    for s in train_data:
        neighbors = sample_generation(s, target_model, target_tokenizer, args)
        
if __name__ == '__main__':
    main()