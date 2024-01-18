from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss

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


def get_idx_unreduced_loss(unreduced_losses, idx):
    batch_size = len(unreduced_losses[0])
    return unreduced_losses[idx // batch_size][idx % batch_size]

def unzip_collate(batch):
        return list(zip(*batch))