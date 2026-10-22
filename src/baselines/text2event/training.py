"""Basic training components. No generation/evaluation or best-checkpoint claim."""
import json
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.baselines.text2event.convert import convert


def load_components(config, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    reference = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'])
    reference.add_special_tokens({'additional_special_tokens': config['additional_special_tokens']})
    if tokenizer.get_vocab() != reference.get_vocab():
        raise ValueError('Saved tokenizer vocabulary differs from pinned revision plus markers')
    if tokenizer.backend_tokenizer.to_str() != reference.backend_tokenizer.to_str():
        raise ValueError('Tokenizer normalizer/serialization differs from pinned tokenizer')
    for marker in config['additional_special_tokens']:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError(f'Invalid marker: {marker}')
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'], revision=config['model_revision'])
    model.resize_token_embeddings(len(tokenizer))
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError('Input embedding vocabulary mismatch')
    if model.get_output_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError('Output embedding vocabulary mismatch')
    return model, tokenizer


class BundleDataset(Dataset):
    def __init__(self, folder, split, tokenizer, max_source_length, max_target_length, stage="full"):
        if stage not in {"full", "trigger"}: raise ValueError("Unknown curriculum stage")
        folder = Path(folder)
        rows = [json.loads(line) for line in (folder / f'{split}.json').read_text().splitlines()]
        gold = [json.loads(line) for line in (folder / f'{split}.gold.jsonl').read_text().splitlines()]
        if len(rows) != len(gold) or not rows:
            raise ValueError('Empty or misaligned bundle')
        self.examples = []
        seen = set()
        for row, sample in zip(rows, gold):
            if sample['id'] in seen or convert(sample) != row:
                raise ValueError(f'Bundle alignment/content mismatch: {sample["id"]}')
            seen.add(sample['id'])
            source = tokenizer(row['text'], truncation=False)
            target_text = row['event']
            if stage == "trigger":
                import copy
                simplified = copy.deepcopy(sample)
                for event in simplified['events']: event['arguments'] = []
                target_text = convert(simplified)['event']
            target = tokenizer(target_text, truncation=False)['input_ids']
            if len(source['input_ids']) > max_source_length or len(target) > max_target_length:
                raise ValueError(f'Length overflow (no truncation): {sample["id"]}')
            self.examples.append({'id': sample['id'], 'input_ids': source['input_ids'],
                                  'attention_mask': source['attention_mask'], 'labels': target})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class BatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([{k: r[k] for k in ['input_ids', 'attention_mask']} for r in rows],
                                   padding=True, return_tensors='pt')
        labels = torch.full((len(rows), max(len(r['labels']) for r in rows)), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            labels[i, :len(row['labels'])] = torch.tensor(row['labels'])
        batch['labels'] = labels
        batch['ids'] = [r['id'] for r in rows]
        return batch


def train_epoch(model, loader, optimizer, scheduler, device, accumulation=1, max_grad_norm=1.0):
    if accumulation < 1 or len(loader) == 0:
        raise ValueError('Require positive accumulation and nonempty loader')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.; count = 0; updates = 0
    for index, batch in enumerate(loader):
        values = {k: v.to(device) for k, v in batch.items() if k != 'ids'}
        # Count actual microbatches in the final accumulation window.
        window_start = (index // accumulation) * accumulation
        window_size = min(accumulation, len(loader) - window_start)
        loss = model(**values).loss
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f'Nonfinite loss at batch {index}')
        (loss / window_size).backward()
        total += loss.item(); count += 1
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            updates += 1
    return {'mean_microbatch_loss': total / count, 'optimizer_updates': updates}
