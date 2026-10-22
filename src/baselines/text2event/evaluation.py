import torch
from src.baselines.text2event.convert import to_unified
from src.baselines.text2event.decoding import TreeConstraint, ConstraintDeadEnd
from src.metrics.evaluator import Evaluator


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
def evaluate_model(model, tokenizer, golds, schema_path, config, device):
    model.eval();predictions=[];diagnostics=[]
    for sample in golds:
        # Only the input fields are passed to generation and span recovery.
        source=' '.join(sample['tokens'])
        encoded=tokenizer(source,return_tensors='pt',truncation=False)
        if encoded['input_ids'].shape[1]>config['max_source_length']: raise ValueError('Evaluation input overflow')
        constraint=TreeConstraint(tokenizer,schema_path,encoded['input_ids'][0].tolist(),
                                  original_tokens=sample['tokens'],max_new_tokens=config['generation_max_new_tokens'])
        issue={'id':sample['id'],'status':'ok','ambiguities':[]}
        try:
            generated=model.generate(**{k:v.to(device) for k,v in encoded.items()},
                max_new_tokens=config['generation_max_new_tokens'],num_beams=1,do_sample=False,
                prefix_allowed_tokens_fn=constraint)
            issue['budget_constrained']=constraint.budget_constrained
            ids=generated[0].tolist()
            if tokenizer.eos_token_id not in ids:
                raise ValueError('generation_limit_without_eos')
            text=tokenizer.decode([i for i in ids if i not in {tokenizer.pad_token_id,tokenizer.eos_token_id}],
                                  skip_special_tokens=False,clean_up_tokenization_spaces=False)
            issue['generated']=text
            pred=to_unified(text,sample['id'],sample['text'],sample['tokens'],
                 ambiguity_policy=config['span_policy'],diagnostics=issue['ambiguities'],
                 surface_normalizer=tokenizer.backend_tokenizer.normalizer.normalize_str)
        except (ValueError,ConstraintDeadEnd) as exc:
            issue.update(status='invalid_prediction',reason=str(exc))
            pred={k:sample[k] for k in ['id','text','tokens']};pred['events']=[]
        predictions.append(pred);diagnostics.append(issue)
    return score_by_id(predictions,golds),predictions,diagnostics
