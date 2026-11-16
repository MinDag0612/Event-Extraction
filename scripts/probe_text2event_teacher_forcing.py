"""Audit fixed TRAIN/DEV targets and teacher-forced predictions without updating weights."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import torch
from src.baselines.text2event.convert import convert_substructure, parse
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator
from src.baselines.text2event.decoding import TreeConstraint
from scripts.compare_text2event_checkpoints import flat_limit


def token_groups(text, offsets, open_id, close_id, eos_id, ids):
    # Character offsets classify the actual serialized target, including multi-piece labels.
    spans=[]
    for match in re.finditer(r'<extra_id_0>\s+([^\s<]+)\s+([^<]+)', text):
        spans.extend([(match.start(1),match.end(1),'label'),
                      (match.start(2),match.end(2),'mention')])
    groups=[]
    for token,(a,b) in zip(ids,offsets):
        if token==eos_id:groups.append('eos')
        elif token in {open_id,close_id}:groups.append('structure')
        else:groups.append(next((kind for start,end,kind in spans if b>start and a<end),'other'))
    return groups


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-checkpoint-epoch',type=int,default=2)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    a.output.mkdir(parents=True,exist_ok=False)
    def save(name,value): (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==a.expected_checkpoint_epoch and state['signature']['stage']=='substructure'
    run=state['signature']['run'];assert run['dataset']=='GENEVA'
    for name,digest in run['files'].items():
        with (a.bundle/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    config=run['config'];model,t=load_components(config,a.bundle/'tokenizer')
    model.load_state_dict(state['model']);del state
    model.to('cuda').eval()
    folder=a.bundle/'GENEVA';op=t.convert_tokens_to_ids('<extra_id_0>');cl=t.convert_tokens_to_ids('<extra_id_1>')
    normalize=lambda s:' '.join(t.backend_tokenizer.normalizer.normalize_str(s).split())
    train=list(map(json.loads,(folder/'train.gold.jsonl').read_text().splitlines()));limit=flat_limit(train)
    save('request.json',{'checkpoint_epoch':a.expected_checkpoint_epoch,'stage':'substructure','training':False,'test_used':False,
        'samples_per_split':8,'seed':421,'config':config,'flat_repeat_limit':limit,
        'note':'Gold prefixes used only for diagnostics, not generation F1; top5 raw logits are unmasked.'})
    summaries=[]
    for split in ['train','dev']:
        rows=list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines()))
        ds=BundleDataset(folder,split,t,config['max_source_length'],config['max_target_length'],
                         stage='substructure',source_prefix=config.get('substructure_source_prefix','span: '))
        indices=sorted(random.Random(421).sample(range(len(rows)),8))
        for index in indices:
            sample=rows[index];example=ds[index];text=convert_substructure(sample)['event']
            encoded=t(text,return_offsets_mapping=True,truncation=False);ids=encoded['input_ids']
            assert ids==example['labels']
            decoded=t.decode([x for x in ids if x!=t.eos_token_id],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            pairs=lambda s:[(x['type'],normalize(x['trigger'])) for x in parse(s)]
            roundtrip=pairs(text)==pairs(decoded)
            groups=token_groups(text,encoded['offset_mapping'],op,cl,t.eos_token_id,ids)
            batch=BatchCollator(t)([example]);values={k:v.to('cuda') for k,v in batch.items() if k!='ids'}
            out=model(**values,use_cache=False);logits=out.logits[0].float().cpu();logs=logits.log_softmax(-1)
            # The same mask is audited twice: original empty-root policy and new nonempty policy.
            constraints={str(allow):TreeConstraint(t,folder/'event.schema',[],original_tokens=sample['tokens'],
                flat=True,flat_repeat_limit=limit,max_new_tokens=config['generation_max_new_tokens'],allow_empty_tree=allow)
                for allow in [True,False]}
            failed={};positions=[];totals={}
            for pos,(gold,group) in enumerate(zip(ids,groups)):
                scores=logits[pos];best=scores.topk(5);allowed={};masked_predictions={}
                for policy,c in constraints.items():
                    if policy in failed:continue
                    try:
                        legal=c(0,torch.tensor([t.pad_token_id]+ids[:pos]));allowed[policy]=gold in legal
                        chosen=legal[scores[legal].argmax().item()]
                        masked_predictions[policy]={'id':chosen,'piece':t.convert_ids_to_tokens(chosen),'correct':chosen==gold}
                        if gold not in legal:failed[policy]={'position':pos,'gold':gold,'reason':'gold token masked'}
                    except ValueError as exc:failed[policy]={'position':pos,'reason':str(exc)}
                nll=-logs[pos,gold].item();correct=int(scores.argmax().item()==gold)
                aggregate=totals.setdefault(group,{'tokens':0,'correct':0,'nll_sum':0.})
                aggregate['tokens']+=1;aggregate['correct']+=correct;aggregate['nll_sum']+=nll
                positions.append({'position':pos,'group':group,'gold_id':gold,'gold_piece':t.convert_ids_to_tokens(gold),
                    'gold_probability':logs[pos,gold].exp().item(),'gold_rank':int((scores>scores[gold]).sum().item())+1,
                    'correct_top1':bool(correct),'gold_allowed':allowed,'masked_top1':masked_predictions,
                    'open_probability':logs[pos,op].exp().item(),'close_probability':logs[pos,cl].exp().item(),
                    'top5':[{'id':i,'piece':t.convert_ids_to_tokens(i),'probability':logs[pos,i].exp().item()} for i in best.indices.tolist()]})
            result={'id':sample['id'],'split':split,'target':text,'decoded':decoded,'roundtrip_pairs_equal':roundtrip,
                'source_tokens':len(example['input_ids']),'target_tokens':len(ids),'within_limits':True,
                'eos_count':ids.count(t.eos_token_id),'pad_in_target':t.pad_token_id in ids,
                'teacher_forced_loss':out.loss.item(),'groups':totals,'mask_failures':failed,'positions':positions}
            save(sample['id'].replace(':','_')+'.json',result)
            summary={k:v for k,v in result.items() if k not in ['positions','target','decoded']};summaries.append(summary)
            print(json.dumps(summary),flush=True)
            del out,logits,logs,values
    save('summary.json',{'samples':summaries,'training':False,'test_used':False})


if __name__=='__main__':main()
