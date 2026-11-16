"""Flat substructure evaluation used to select the stage-1 checkpoint."""
from src.baselines.text2event.evaluation_trace import SampleTimer
import torch
from src.baselines.text2event.convert import convert_substructure, parse
from src.baselines.text2event.upstream_decoding import UpstreamConstraint
from src.baselines.text2event.logit_rules import EmptyRootPenalty
from src.baselines.text2event.decoding import TreeConstraint


def leaf_set(sequence, normalize):
    records = parse(sequence)
    if any(record['roles'] for record in records):
        raise ValueError('Substructure must be flat')
    return {(record['type'], normalize(record['trigger'])) for record in records}


@torch.no_grad()
def evaluate_substructure(model, tokenizer, samples, schema_path, config, device, observer=None):
    model.eval(); tp = fp = fn = 0; issues = []
    penalty=EmptyRootPenalty(tokenizer.convert_tokens_to_ids('<extra_id_0>'),
        tokenizer.convert_tokens_to_ids('<extra_id_1>'),tokenizer.pad_token_id,config.get('empty_root_penalty',0.))
    normalize = tokenizer.backend_tokenizer.normalizer.normalize_str
    prefix = config.get('substructure_source_prefix', 'span: ')
    prefix_length = len(tokenizer.encode(prefix, add_special_tokens=False))
    for index, sample in enumerate(samples, 1):
        timer=SampleTimer(observer,sample['id'],index,len(samples))
        timer.start('encode')
        text = prefix+' '.join(sample['tokens'])
        encoded = tokenizer(text, return_tensors='pt', truncation=False)
        if encoded['input_ids'].shape[1] > config['max_source_length']:
            raise ValueError('Substructure evaluation input overflow')
        timer.end('encode');timer.start('constraint')
        if config.get('substructure_constraint_decoder') == 'project-constrained':
            limits = config.get('constraint_repetition_limits', {}).get(schema_path.parent.name, {})
            constraint = TreeConstraint(tokenizer, schema_path, [], original_tokens=sample['tokens'],
                flat=True, flat_repeat_limit=limits.get('flat_repeat_limit', 1),
                max_new_tokens=config['generation_max_new_tokens'],
                allow_empty_tree=config.get('constraint_allow_empty_tree', True))
        else:
            constraint = UpstreamConstraint(tokenizer,schema_path,
                encoded['input_ids'][0].tolist()[prefix_length:],flat=True)
        timer.end('constraint')
        issue = {'id':sample['id'], 'status':'ok','empty_root_penalty':penalty.penalty}
        timer.start('generate')
        try:
            output = model.generate(**{k:v.to(device) for k,v in encoded.items()},
                max_new_tokens=config['generation_max_new_tokens'],num_beams=1,
                do_sample=False,use_cache=True,prefix_allowed_tokens_fn=timer.wrap(constraint),logits_processor=[penalty])[0].tolist()
            timer.end('generate')
            sequence = tokenizer.decode([i for i in output if i not in {tokenizer.pad_token_id,tokenizer.eos_token_id}],
                skip_special_tokens=False,clean_up_tokenization_spaces=False)
            issue['generated'] = sequence
            issue['generated_tokens'] = len(output)-1
            issue['budget_constrained'] = constraint.budget_constrained
            if tokenizer.eos_token_id not in output:
                raise ValueError('generation_limit_without_eos')
            predicted = leaf_set(sequence, normalize)
        except (ValueError, KeyError) as exc:
            issue.update(status='invalid_prediction',reason=str(exc)); predicted=set()
        # Gold is used only for scoring, never to select legal next tokens.
        gold = leaf_set(convert_substructure(sample)['event'], normalize)
        tp += len(predicted & gold); fp += len(predicted-gold); fn += len(gold-predicted)
        timer.finish(issue)
        issues.append(issue)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'precision':precision,'recall':recall,'f1':f1}, issues
