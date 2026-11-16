"""TRAIN-only learning diagnostics; no rules derived from held-out gold."""
import torch
import torch.nn.functional as F
from src.baselines.text2event.convert import records, parse


def f1_counts(tp, fp, fn):
    return {'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'gold_count':tp+fn,'predicted_count':tp+fp}


def component_scores(samples, issues, types, roles, normalize):
    """Split flat label/surface scores. Shared type/role labels remain ambiguous."""
    ambiguous=set(types)&set(roles)
    counts={'type_trigger':[0,0,0], 'role_argument':[0,0,0]}
    indexed={issue['id']:issue for issue in issues}
    for sample in samples:
        gold={'type_trigger':set(),'role_argument':set()}
        for record in records(sample):
            gold['type_trigger'].add((record['type'],normalize(record['trigger'])))
            gold['role_argument'].update((role,normalize(text)) for role,text in record['roles'])
        issue=indexed[sample['id']];predicted=set()
        if issue['status']=='ok':
            predicted={(r['type'],normalize(r['trigger'])) for r in parse(issue['generated'])}
        for name,labels in [('type_trigger',set(types)),('role_argument',set(roles))]:
            pred={pair for pair in predicted if pair[0] in labels-ambiguous}
            target={pair for pair in gold[name] if pair[0] not in ambiguous}
            for i,v in enumerate([len(pred & target),len(pred-target),len(target-pred)]):counts[name][i]+=v
    return {**{name:f1_counts(*values) for name,values in counts.items()},
            'excluded_ambiguous_type_role_labels':sorted(ambiguous)}


def gate_passed(scores, threshold):
    return all(scores[name]['f1']>=threshold and scores[name].get('gold_count',1)>0 for name in ['type_trigger','role_argument'])


def target_token_groups(ids, tokenizer, types, roles, flat=False):
    """Offline token labels for telemetry; shared type/role labels are explicit."""
    op=tokenizer.convert_tokens_to_ids('<extra_id_0>');cl=tokenizer.convert_tokens_to_ids('<extra_id_1>')
    groups=['mention']*len(ids);depth=0;i=0
    encoded={name:tokenizer.encode(name,add_special_tokens=False) for name in set(types)|set(roles)}
    while i<len(ids):
        token=ids[i]
        if token==tokenizer.eos_token_id:groups[i]='eos';i+=1;continue
        if token in {op,cl}:
            groups[i]='structure';depth+=1 if token==op else -1;i+=1
            if token==op and depth in ({2} if flat else {2,3}):
                labels=(set(types)|set(roles)) if flat else set(types if depth==2 else roles)
                matches=[name for name in labels if ids[i:i+len(encoded[name])]==encoded[name]]
                if matches:
                    name=max(matches,key=lambda n:len(encoded[n]));n=len(encoded[name])
                    group=('ambiguous_label' if name in types and name in roles else 'type_label' if name in types else 'role_label') if flat else ('type_label' if depth==2 else 'role_label')
                    groups[i:i+n]=[group]*n;i+=n
            continue
        i+=1
    return groups


@torch.no_grad()
def teacher_probe(model, tokenizer, examples, device, max_errors=12, label_schema=None, flat=False):
    """Deterministic teacher-forced loss/token accuracy; not free-generation F1."""
    model.eval();result=[];loss_sum=0.;tokens=correct=0;group_counts={}
    op=tokenizer.convert_tokens_to_ids('<extra_id_0>');cl=tokenizer.convert_tokens_to_ids('<extra_id_1>')
    for example in examples:
        ids=torch.tensor([example['input_ids']],device=device)
        attention=torch.tensor([example['attention_mask']],device=device)
        target=torch.tensor(example['labels'],device=device)
        output=model(input_ids=ids,attention_mask=attention,labels=target.unsqueeze(0))
        logits=output.logits[0].float()
        losses=F.cross_entropy(logits,target,reduction='none');pred=logits.argmax(-1)
        if label_schema is not None:
            groups=target_token_groups(example['labels'],tokenizer,*label_schema,flat=flat)
            for group in set(groups):
                positions=[i for i,g in enumerate(groups) if g==group]
                values=group_counts.setdefault(group,{'tokens':0,'correct':0,'loss_sum':0.})
                values['tokens']+=len(positions)
                values['correct']+=int((pred[positions]==target[positions]).sum())
                values['loss_sum']+=float(losses[positions].sum())
        errors=(pred!=target).nonzero().flatten().tolist()
        loss_sum+=losses.sum().item();tokens+=len(target);correct+=(pred==target).sum().item()
        row={'id':example['id'],'mean_token_loss':losses.mean().item(),
             'wrong_tokens':[{'position':i,'expected_id':int(target[i]),'predicted_id':int(pred[i]),
                 'expected':tokenizer.decode([int(target[i])]),'predicted':tokenizer.decode([int(pred[i])]),
                 'loss':float(losses[i])} for i in errors[:max_errors]]}
        if len(target)>1 and int(target[0])==op:
            row['root_choice']={'expected_next_id':int(target[1]),
                'close_minus_open_logit':float(logits[1,cl]-logits[1,op]),
                'p_close_given_open_or_close':float(logits[1,[op,cl]].softmax(0)[1])}
        result.append(row)
    return {'mean_token_loss':loss_sum/tokens if tokens else 0.,
            'token_accuracy':correct/tokens if tokens else 0.,'examples':result,
            'token_groups':{g:{**v,'accuracy':v['correct']/v['tokens'],
                'mean_token_loss':v['loss_sum']/v['tokens']} for g,v in group_counts.items()}}
