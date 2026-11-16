from src.baselines.text2event.evaluation_trace import SampleTimer
import torch
from types import SimpleNamespace
from src.baselines.text2event.upstream_decoding import UpstreamConstraint
from src.baselines.text2event.convert import to_unified
from src.baselines.text2event.decoding import TreeConstraint, ConstraintDeadEnd
from src.metrics.evaluator import Evaluator
from src.baselines.text2event.logit_rules import EmptyRootPenalty


def score_by_id(predictions, golds):
    def index(rows):
        result={}
        for row in rows:
            if row['id'] in result: raise ValueError('Duplicate evaluation ID')
            result[row['id']]=row
        return result
    pred=index(predictions); gold=index(golds)
    if pred.keys()!=gold.keys(): raise ValueError('Prediction/gold IDs differ')
    ids=sorted(gold)
    return Evaluator().evaluate_on_all_sample([pred[i] for i in ids],[gold[i] for i in ids])


@torch.no_grad()
def evaluate_model(model, tokenizer, golds, schema_path, config, device, observer=None):
    model.eval();predictions=[];diagnostics=[]
    penalty=EmptyRootPenalty(tokenizer.convert_tokens_to_ids('<extra_id_0>'),
        tokenizer.convert_tokens_to_ids('<extra_id_1>'), tokenizer.pad_token_id,
        config.get('empty_root_penalty',0.))
    for index, sample in enumerate(golds, 1):
        timer=SampleTimer(observer,sample['id'],index,len(golds))
        timer.start('encode')
        # Only the input fields are passed to generation and span recovery.
        source_prefix=config.get('full_source_prefix', '')
        source=source_prefix+' '.join(sample['tokens'])
        encoded=tokenizer(source,return_tensors='pt',truncation=False)
        if encoded['input_ids'].shape[1]>config['max_source_length']: raise ValueError('Evaluation input overflow')
        timer.end('encode');timer.start('constraint')
        limits=config.get('constraint_repetition_limits', {}).get(schema_path.parent.name, {})
        if config.get('constraint_decoder') == 'unconstrained':
            constraint=SimpleNamespace(repetition_policy='unconstrained', event_repeat_limit=None,
                                       argument_repeat_limit=None, budget_constrained=False)
        elif config.get('constraint_decoder') in {'upstream','upstream-compatible','upstream-exact'}:
            prefix_length=len(tokenizer.encode(source_prefix, add_special_tokens=False))
            constraint=UpstreamConstraint(tokenizer,schema_path,encoded['input_ids'][0].tolist()[prefix_length:],
                                          resolve_label_prefixes=config.get('constraint_decoder') != 'upstream-exact')
        else:
            constraint=TreeConstraint(tokenizer,schema_path,encoded['input_ids'][0].tolist(),
                                      original_tokens=sample['tokens'],max_new_tokens=config['generation_max_new_tokens'],
                                      repetition_policy=config.get('constraint_repetition_policy', 'source-occurrence'),
                                      event_repeat_limit=limits.get('event_repeat_limit', 1),
                                      argument_repeat_limit=limits.get('argument_repeat_limit', 1),
                                      allow_empty_tree=config.get('constraint_allow_empty_tree', True))
        timer.end('constraint')
        issue={'id':sample['id'],'status':'ok','ambiguities':[],
               'constraint_policy':constraint.repetition_policy,
               'decoder':config.get('constraint_decoder','project-constrained'),
               'empty_root_penalty':penalty.penalty,
               'event_repeat_limit':constraint.event_repeat_limit,
               'argument_repeat_limit':constraint.argument_repeat_limit}
        timer.start('generate')
        try:
            generated=model.generate(**{k:v.to(device) for k,v in encoded.items()},
                max_new_tokens=config['generation_max_new_tokens'],num_beams=1,do_sample=False,
                use_cache=True,prefix_allowed_tokens_fn=timer.wrap(constraint),logits_processor=[penalty])
            timer.end('generate')
            issue['budget_constrained']=constraint.budget_constrained
            ids=generated[0].tolist()
            text=tokenizer.decode([i for i in ids if i not in {tokenizer.pad_token_id,tokenizer.eos_token_id}],
                                  skip_special_tokens=False,clean_up_tokenization_spaces=False)
            issue['generated']=text
            # Preserve failed raw generations: invalid-as-empty is a scorer
            # fallback, not evidence that the model chose an empty root.
            issue['generated_tokens']=len(ids)-1
            if tokenizer.eos_token_id not in ids:
                raise ValueError('generation_limit_without_eos')
            pred=to_unified(text,sample['id'],sample['text'],sample['tokens'],
                 ambiguity_policy=config['span_policy'],diagnostics=issue['ambiguities'],
                 surface_normalizer=tokenizer.backend_tokenizer.normalizer.normalize_str)
        except (ValueError,KeyError,ConstraintDeadEnd) as exc:
            issue.update(status='invalid_prediction',reason=str(exc))
            pred={k:sample[k] for k in ['id','text','tokens']};pred['events']=[]
        timer.finish(issue)
        predictions.append(pred);diagnostics.append(issue)
    return score_by_id(predictions,golds),predictions,diagnostics
