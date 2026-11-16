"""Audit lengths and replay sampled gold prefixes for both upstream-style stages."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.baselines.text2event.convert import convert, convert_substructure
from src.baselines.text2event.upstream_decoding import UpstreamConstraint


def audit(bundle, config):
    tokenizer=AutoTokenizer.from_pretrained(bundle/'tokenizer',local_files_only=True)
    report={}
    for dataset in ['BKEE','GENEVA','PHEE','VHE']:
        folder=bundle/dataset;report[dataset]={}
        for split in ['train','dev','test']:
            rows=list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines()))
            stages={}
            for stage,convert_fn,prefix in [('substructure',convert_substructure,config['substructure_source_prefix']),
                                           ('full',convert,config['full_source_prefix'])]:
                max_source=max_target=0;overflow=[];targets=[]
                for row in rows:
                    converted=convert_fn(row)
                    source_length=len(tokenizer.encode(prefix+converted['text']))
                    ids=tokenizer.encode(converted['event']);targets.append(ids)
                    max_source=max(max_source,source_length);max_target=max(max_target,len(ids))
                    if source_length>config['max_source_length'] or len(ids)>config['max_target_length']:
                        overflow.append(row['id'])
                # Include the longest target as well as initial examples, for all splits.
                chosen=sorted(set(list(range(min(4,len(rows))))+[max(range(len(rows)),key=lambda i:len(targets[i]))]))
                failures=[];passed=0
                for index in chosen:
                    row=rows[index]
                    encoded=tokenizer.encode(prefix+' '.join(row['tokens']))
                    offset=len(tokenizer.encode(prefix,add_special_tokens=False))
                    constraint=UpstreamConstraint(tokenizer,folder/'event.schema',encoded[offset:],flat=stage=='substructure')
                    generated=[tokenizer.pad_token_id]
                    try:
                        for position,token in enumerate(targets[index]):
                            if token not in constraint(0,torch.tensor(generated)):
                                raise ValueError(f'Token {token} rejected at position {position}')
                            generated.append(token)
                    except (ValueError,KeyError) as exc:
                        failures.append({'id':row['id'],'reason':str(exc)})
                    else:passed+=1
                stages[stage]={'samples':len(rows),'max_source_tokens':max_source,'max_target_tokens':max_target,
                    'length_overflows':len(overflow),'overflow_examples':overflow[:10],
                    'replay_checked':len(chosen),'replay_passed':passed,'replay_failures':failures}
            report[dataset][split]=stages
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/text2event/preparation.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.bundle,json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({d:{s:{k:{'overflows':v['length_overflows'],'replay':f"{v['replay_passed']}/{v['replay_checked']}"} for k,v in stages.items()} for s,stages in splits.items()} for d,splits in result.items()}))
