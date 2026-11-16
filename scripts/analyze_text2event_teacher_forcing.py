"""Summarize decision boundaries from saved teacher-forcing JSON; no model or GPU."""
import argparse
from collections import Counter
import json
from pathlib import Path


def decisions(row, policy='False'):
    """Classify gold-path decisions; keep standalone SentencePiece spacing distinct."""
    ps=row['positions'];events=[];leaf=0
    def add(kind,p,**extra):
        predicted=p['masked_top1'][policy]
        events.append({'kind':kind,'position':p['position'],'gold':p['gold_piece'],
            'predicted':predicted['piece'],'correct':predicted['correct'],
            'gold_rank_raw':p['gold_rank'],'gold_probability_raw':p['gold_probability'],
            'open_probability_raw':p['open_probability'],'close_probability_raw':p['close_probability'],**extra})
    for i,p in enumerate(ps):
        previous=ps[i-1] if i else None
        if p['group']=='label' and (previous is None or previous['group']!='label'):
            leaf+=1;add('label_start',p,leaf=leaf)
            end=i
            while end<len(ps) and ps[end]['group']=='label':end+=1
            add('label_complete_gold_path',p,leaf=leaf,
                all_tokens_correct=all(x['masked_top1'][policy]['correct'] for x in ps[i:end]))
        if p['group']=='mention' and (previous is None or previous['group']!='mention'):
            add('mention_start',p,leaf=leaf)
            content=i
            while content<len(ps) and ps[content]['group']=='mention' and ps[content]['gold_piece']=='▁':content+=1
            if content<len(ps) and ps[content]['group']=='mention':
                add('mention_first_content_piece',ps[content],leaf=leaf)
        if p['group']=='structure':
            if previous and previous['group']=='mention':add('mention_end_close_leaf',p,leaf=leaf)
            elif p['gold_piece']=='<extra_id_0>' and i==1:add('root_first_leaf',p)
            elif previous and previous['gold_piece']=='<extra_id_1>':
                add('continue_next_leaf' if p['gold_piece']=='<extra_id_0>' else 'finish_root',p,leaf=leaf)
    return events


def analyze(folder):
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob('GENEVA*.json'))]
    if len(rows)!=16:raise ValueError('Expected complete fixed 8TRAIN/8DEV diagnostic')
    result={'scope':'gold-prefix diagnostic only; not free-generation accuracy or F1','samples':[],'splits':{}}
    for row in rows:
        result['samples'].append({'id':row['id'],'split':row['split'],'decisions':decisions(row)})
    for split in ['train','dev']:
        samples=[r for r in result['samples'] if r['split']==split];stats={}
        for kind in sorted({e['kind'] for r in samples for e in r['decisions']}):
            events=[e for r in samples for e in r['decisions'] if e['kind']==kind]
            correct=sum(e.get('all_tokens_correct',e['correct']) for e in events)
            stats[kind]={'correct':correct,'count':len(events),'accuracy':correct/len(events),
                         'predicted_piece_counts':dict(Counter(e['predicted'] for e in events))}
        result['splits'][split]=stats
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.input);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result['splits'],ensure_ascii=False,indent=2))
