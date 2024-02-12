from argparse import ArgumentParser
import torch

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
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
    
def main():
    args = get_cli_args()
    target_model, target_tokenizer, = load_model(args['target_model'], path=args['target_path'])

if __name__ == '__main__':
    main()