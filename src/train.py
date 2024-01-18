import pandas
import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from pathlib import Path
from MIA.load_utils import create_new_dir
from argparse import ArgumentParser
from MIA.load_utils import load_model
from datasets import load_dataset

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default = 5)
    parser.add_argument('--val_only', action='store_true')
    args = parser.parse_args()
    return vars(args)

def get_data(path, split):
    CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'

    if 'newsSpace' in path:
        df = pandas.read_csv(path, names=['id', 'category', 'text'], header=0, index_col='id', encoding='utf-8')
        train_data = df['text'].astype(str).to_list()
    elif 'sentiment140' in path:
        dataset = load_dataset("sentiment140", split='train', cache_dir=CACHE_DIR)
        if split == 'train':
            train_data = dataset['text'][:150000]
        else:
            train_data = dataset['text'][150000:300000]
    elif 'wikitext' in path:
        dataset = load_dataset("wikitext", 'wikitext-103-raw-v1', split='train', cache_dir=CACHE_DIR)
        dataset = [row for row in dataset['text'] if len(row) > 0]
        if split == 'train':
            train_data = dataset[:100000]
        else:
            train_data = dataset[100000:200000]
    return train_data

def main():
    args = get_cli_args()
    do_train = not args['val_only']
    output_dir = create_new_dir('checkpoints/gpt2')
    with open(os.path.join(output_dir, 'args.json'), mode='w') as f:
        json.dump(args, f)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data = get_data(args['train_path'], 'train')
    val_data = get_data(args['val_path'], 'val')

    model, tokenizer = load_model(args['model_name'])
    # tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=CACHE_DIR)
    # tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        tokens = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=150)
        labels = torch.tensor([
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(tokens["attention_mask"], tokens["input_ids"])]
        ])
        tokens['labels'] = labels
        return tokens

    if 'newsSpace' in args['train_path']:
        train_dl = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, shuffle=True)
    else:
        train_dl = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, shuffle=False)
    val_dl = DataLoader(val_data, batch_size=32, collate_fn=collate_fn, shuffle=False)

    # model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=CACHE_DIR)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    torch.cuda.empty_cache()
    for epoch in range(args['epochs']) if do_train else range(1):
        if do_train:
            model.train()
            progress_bar = tqdm(range(len(train_dl)))
            for batch in train_dl:
                batch = batch.to(device)
                loss = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels']).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        model.eval()
        val_bar = tqdm(range(len(val_dl)))
        losses = []
        for batch in val_dl:
            batch = batch.to(device)
            with torch.no_grad():
                outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss[shift_labels.view(-1) != -100]
                losses.append(loss)
            val_bar.update(1)
        mean_loss = torch.mean(torch.cat(losses))
        print(f"mean loss = {mean_loss}")
        perplexity = torch.exp(mean_loss)
        print(f"perplexity = {perplexity}")

        if do_train:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'epoch-{epoch}_perplexity-{perplexity:0,.4f}'))

if __name__ == "__main__":
    main()