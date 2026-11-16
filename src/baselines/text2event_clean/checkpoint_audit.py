"""Measure gold-prefix decisions with the exact decoder stored in a clean checkpoint."""
from collections import Counter
import json
import random
import torch
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator
from src.baselines.text2event.convert import convert_substructure
from src.baselines.text2event.upstream_decoding import UpstreamConstraint
from src.baselines.text2event_clean.preflight import bundle_hashes
from scripts.probe_text2event_teacher_forcing import token_groups
from scripts.analyze_text2event_teacher_forcing import decisions


def validate_checkpoint(state, config, bundle, expected_epoch):
    """Require the clean epoch/stage, model config and identical data artifacts."""
    signature=state['signature'];run=signature['run']
    if signature['stage']!='substructure' or state['epoch']!=expected_epoch:
        raise ValueError('Unexpected checkpoint epoch/stage')
    if run['implementation'] not in {'text2event-clean-pilot-v1','text2event-clean-lr-ablation-v1'} or run['dataset']!='GENEVA':
        raise ValueError('Expected clean GENEVA pilot checkpoint')
    if run['config']!=config or run['files']!=bundle_hashes(bundle):
        raise ValueError('Checkpoint config/data differs')


@torch.no_grad()
def run(bundle, config, checkpoint, output, expected_epoch):
    """Read frozen weights and score16fixed gold-prefix paths without optimizer updates."""
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(state,config,bundle,expected_epoch)
    save('request.json',{'kind':'clean-checkpoint-gold-prefix-audit','checkpoint_epoch':state['epoch'],
        'checkpoint_bytes':checkpoint.stat().st_size,'config':config,'samples_per_split':8,
        'selection_seed':421,'trained':False,'test_used':False,'decoder_changed':False})
    model,t=load_components(config,bundle/'tokenizer');model.load_state_dict(state['model']);del state
    model.to('cuda').eval()
    folder=bundle/'GENEVA';policy=config['substructure_constraint_decoder']
    op=t.convert_tokens_to_ids('<extra_id_0>');cl=t.convert_tokens_to_ids('<extra_id_1>')
    summaries=[];decision_samples=[]
    for split in ['train','dev']:
        rows=list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines()))
        ds=BundleDataset(folder,split,t,config['max_source_length'],config['max_target_length'],
                         'substructure',config['substructure_source_prefix'])
        indices=sorted(random.Random(421).sample(range(len(rows)),8))
        for index in indices:
            item=ds[index];text=convert_substructure(rows[index])['event']
            encoded=t(text,return_offsets_mapping=True,truncation=False);ids=item['labels']
            if encoded['input_ids']!=ids:raise ValueError('Target token IDs differ')
            groups=token_groups(text,encoded['offset_mapping'],op,cl,t.eos_token_id,ids)
            prefix_length=len(t.encode(config['substructure_source_prefix'],add_special_tokens=False))
            constraint=UpstreamConstraint(t,folder/'event.schema',item['input_ids'][prefix_length:],flat=True)
            batch=BatchCollator(t)([item]);values={k:v.to('cuda') for k,v in batch.items() if k!='ids'}
            out=model(**values,use_cache=False);loss=out.loss.item();logits=out.logits[0].float().cpu()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite checkpoint logits')
            logs=logits.log_softmax(-1);positions=[];totals={};mask_failure=None
            for pos,(gold,group) in enumerate(zip(ids,groups)):
                scores=logits[pos];masked={};allowed={}
                if mask_failure is None:
                    try:
                        legal=sorted(set(constraint(0,torch.tensor([t.pad_token_id]+ids[:pos]))))
                        allowed[policy]=gold in legal
                        chosen=legal[scores[legal].argmax().item()]
                        masked[policy]={'id':chosen,'piece':t.convert_ids_to_tokens(chosen),'correct':chosen==gold}
                        if gold not in legal:mask_failure={'position':pos,'reason':'gold token masked'}
                    except (ValueError,KeyError) as error:mask_failure={'position':pos,'reason':str(error)}
                stats=totals.setdefault(group,{'tokens':0,'raw_correct':0,'masked_correct':0,'nll_sum':0.})
                stats['tokens']+=1;stats['raw_correct']+=int(scores.argmax().item()==gold)
                stats['masked_correct']+=int(masked.get(policy,{}).get('correct',False))
                stats['nll_sum']-=logs[pos,gold].item()
                top=scores.topk(5).indices.tolist()
                positions.append({'position':pos,'group':group,'gold_piece':t.convert_ids_to_tokens(gold),
                    'gold_probability':logs[pos,gold].exp().item(),'gold_rank':int((scores>scores[gold]).sum().item())+1,
                    'masked_top1':masked,'gold_allowed':allowed,'open_probability':logs[pos,op].exp().item(),
                    'close_probability':logs[pos,cl].exp().item(),
                    'close_probability_given_open_or_close':scores[[op,cl]].softmax(0)[1].item(),
                    'raw_top5':[{'piece':t.convert_ids_to_tokens(k),'probability':logs[pos,k].exp().item()} for k in top]})
            row={'id':item['id'],'split':split,'target':text,'teacher_forced_loss':loss,
                 'groups':totals,'mask_failure':mask_failure,'positions':positions}
            save(item['id'].replace(':','_')+'.json',row)
            if mask_failure is None:
                decision_samples.append({'id':item['id'],'split':split,'decisions':decisions(row,policy)})
            summary={k:v for k,v in row.items() if k not in ['target','positions']}
            summary['root']=positions[1]
            summaries.append(summary);print(json.dumps(summary),flush=True)
            del out,logits,logs,values
    stats={}
    for split in ['train','dev']:
        selected=[s for s in decision_samples if s['split']==split];stats[split]={}
        for kind in sorted({e['kind'] for s in selected for e in s['decisions']}):
            events=[e for s in selected for e in s['decisions'] if e['kind']==kind]
            stats[split][kind]={'correct':sum(e.get('all_tokens_correct',e['correct']) for e in events),
                'count':len(events),'predicted_piece_counts':dict(Counter(e['predicted'] for e in events))}
    save('summary.json',{'samples':summaries,'decision_stats':stats,'decoder':policy,'trained':False,
        'scope':'gold-prefix token decisions, not free-generation F1','test_used':False})
    save('decision-audit.json',{'samples':decision_samples,'splits':stats})
    save('completion.json',{'status':'complete','samples':16,'trained':False,'test_used':False})
