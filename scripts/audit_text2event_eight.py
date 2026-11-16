"""Gold prefix audit: annotation is checked offline, never used to constrain prediction."""
import torch
from src.baselines.text2event.convert import convert,convert_substructure,parse
from src.baselines.text2event.upstream_decoding import UpstreamConstraint


def audit_selected(tokenizer,splits,schema,config):
    failures=[];checked={}
    markers={m:tokenizer.encode(m,add_special_tokens=False) for m in ['<extra_id_0>','<extra_id_1>']}
    if any(len(v)!=1 or v[0]==tokenizer.unk_token_id for v in markers.values()) or len({tuple(v) for v in markers.values()})!=2:
        failures.append({'reason':'Tree markers must be distinct single tokens'})
    for split,rows in splits.items():
        checked[split]={}
        for stage,converter in [('substructure',convert_substructure),('full',convert)]:
            prefix=config['substructure_source_prefix' if stage=='substructure' else 'full_source_prefix']
            count=0
            for row in rows:
                record=converter(row);source=tokenizer.encode(prefix+record['text']);target=tokenizer.encode(record['event'])
                try:
                    if len(source)>config['max_source_length'] or len(target)>config['max_target_length']:raise ValueError('Length overflow')
                    decoded=tokenizer.decode([i for i in target if i not in {tokenizer.pad_token_id,tokenizer.eos_token_id}],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    parse(decoded)
                    off=len(tokenizer.encode(prefix,add_special_tokens=False))
                    c=UpstreamConstraint(tokenizer,schema,source[off:],flat=stage=='substructure')
                    generated=[tokenizer.pad_token_id]
                    for pos,token in enumerate(target):
                        if token not in c(0,torch.tensor(generated)):raise ValueError(f'Gold token rejected at {pos}: {token}')
                        generated.append(token)
                except (ValueError,KeyError) as exc:failures.append({'split':split,'stage':stage,'id':row['id'],'reason':str(exc)})
                else:count+=1
            checked[split][stage]={'total':len(rows),'passed':count}
    return {'tokenizer_class':type(tokenizer).__name__,'is_fast':tokenizer.is_fast,
            'markers':markers,'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id,
            'checked':checked,'failures':failures,
            'slow_tokenizer_parity':'Not established; use_fast=False still selects fast backend in this installed Transformers version.'}
