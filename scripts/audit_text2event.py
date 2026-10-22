"""Measure tokenizer and deterministic parser behavior, without training a model."""
import argparse
import json
import math
from pathlib import Path
from src.baselines.text2event.convert import parse, to_unified
from src.metrics.evaluator import Evaluator

MODEL='google/mt5-base'
REVISION='2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f'


def audit(root, output):
    from transformers import AutoTokenizer
    import transformers
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    # mT5 checkpoints do not necessarily register the T5 sentinels by default.
    markers=['<extra_id_0>','<extra_id_1>']
    added=tokenizer.add_special_tokens({'additional_special_tokens':markers})
    for marker in markers:
        ids=tokenizer.encode(marker,add_special_tokens=False)
        if len(ids)!=1 or ids[0]==tokenizer.unk_token_id:
            raise ValueError('Tree marker must be a single known token')
    tokenizer.save_pretrained(root/'tokenizer')
    report={'model':MODEL,'revision':REVISION,'transformers':transformers.__version__,
            'added_tokens':added,'tokenizer_length':len(tokenizer),
            'span_policy':'leftmost unused trigger per type/surface, then reuse; nearest argument to trigger center; leftmost tie',
            'datasets':{}, 'model_inference_run':False}
    all_input=[];all_target=[]
    for name in ['BKEE','GENEVA','PHEE','VHE']:
        report['datasets'][name]={}
        for split in ['train','dev','test']:
            rows=[json.loads(l) for l in (root/name/f'{split}.json').read_text().splitlines()]
            gold=[json.loads(l) for l in (root/name/f'{split}.gold.jsonl').read_text().splitlines()]
            metric=Evaluator(); length_in=[];length_out=[]; unknown=0; changed=0; failures=[];ambiguities=0
            for batch_start in range(0,len(rows),256):
                batch=rows[batch_start:batch_start+256]
                sources=tokenizer([r['text'] for r in batch],truncation=False)['input_ids']
                targets=tokenizer([r['event'] for r in batch],truncation=False)['input_ids']
                for j,(r,ins,outs) in enumerate(zip(batch,sources,targets)):
                    g=gold[batch_start+j]; length_in.append(len(ins)); length_out.append(len(outs))
                    unknown+=int(tokenizer.unk_token_id in ins or tokenizer.unk_token_id in outs)
                    # Remove only EOS/PAD, never tree sentinels.
                    decoded=tokenizer.decode([i for i in outs if i not in {tokenizer.eos_token_id,tokenizer.pad_token_id}],
                                              skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    issues=[]
                    try:
                        changed+=int(parse(decoded)!=parse(r['event']))
                        pred=to_unified(decoded,g['id'],g['text'],g['tokens'],ambiguity_policy='nearest-trigger',diagnostics=issues,
                                           surface_normalizer=tokenizer.backend_tokenizer.normalizer.normalize_str)
                    except ValueError as exc:
                        failures.append({'id':g['id'],'reason':str(exc)})
                        pred=dict(g,events=[])
                    ambiguities+=bool(issues)
                    metric.evaluate_on_sample(pred,g)
            def stats(values):
                ordered=sorted(values)
                return {'max':max(values),'p99':ordered[math.ceil(.99*len(values))-1],
                        'over_512':sum(x>512 for x in values),'over_1024':sum(x>1024 for x in values)}
            report['datasets'][name][split]={'samples':len(rows),'input':stats(length_in),'target':stats(length_out),
                'samples_with_unk':unknown,'surface_changed_after_tokenization':changed,
                'parse_failures':len(failures),'ambiguous_samples':ambiguities,
                'gold_target_decode_scores':{k:m.compute() for k,m in [('TI',metric.ti),('TC',metric.tc),('AI',metric.ai),('AC',metric.ac)]}}
            (root/name/f'{split}.parser-failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
            all_input.extend(length_in);all_target.extend(length_out)
            print(name,split,report['datasets'][name][split],flush=True)
    config={'model_name':MODEL,'revision':REVISION,'tokenizer_path':str(root/'tokenizer'),
            'max_source_length':math.ceil(max(all_input)/64)*64,
            'max_target_length':math.ceil(max(all_target)/64)*64,
            'truncation':False,'ambiguity_policy':'nearest-trigger',
            'resize_token_embeddings_to_tokenizer':True,
            'surface_normalizer':'saved tokenizer backend normalizer',
            'length_policy':'sized from all splits for coverage only, not model selection',
            'ready_for_training':False}
    report['config']=config
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (root/'preparation-config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.input,a.output)
