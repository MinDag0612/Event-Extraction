"""Loader audit, real forward/backward smoke test, and sequential training."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.oneie.runtime import load_runtime


class SafeCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, logits, target):
        if not (target != self.ignore_index).any():
            return logits.sum() * 0
        return super().forward(logits, target)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['BKEE','GENEVA','RAMS','VHE'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args=parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    IEDataset, Config, OneIE, generate_vocabs, load_valid_patterns = load_runtime()
    from scorer import score_graphs
    root=Path(__file__).resolve().parents[2]
    config=Config.from_json_file(root/f'configs/oneie/{args.dataset}.json')
    config.use_gpu=torch.cuda.is_available() and not args.cpu
    if args.batch_size is not None:
        config.batch_size=args.batch_size
    tokenizer=BertTokenizer.from_pretrained(root/"data/oneie/backbone", do_lower_case=False)
    sets=[IEDataset(getattr(config,f'{s}_file'), max_length=config.max_length,
                    gpu=config.use_gpu, symmetric_relations=[]) for s in ('train','dev','test')]
    # Label names follow upstream's union protocol; valid patterns use train only.
    vocabs=generate_vocabs(sets)
    for dataset in sets:
        dataset.numberize(tokenizer,vocabs)
    patterns=load_valid_patterns(config.valid_pattern_path,vocabs)
    output=Path(config.log_path); output.mkdir(parents=True,exist_ok=True)
    (output/'vocabs.json').write_text(json.dumps(vocabs,indent=2),encoding='utf-8')
    print('LOADER PASS', [len(s) for s in sets], flush=True)
    model=OneIE(config,vocabs,patterns)
    model.load_bert(str(root/"data/oneie/backbone"),cache_dir=config.bert_cache_dir)
    for name in ('entity','event','mention','relation','role'):
        setattr(model,f'{name}_criteria',SafeCrossEntropy())
    if config.use_gpu: model.cuda()
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if n.startswith('bert')], 'lr':config.bert_learning_rate},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('bert')], 'lr':config.learning_rate}],weight_decay=config.weight_decay)
    loader=DataLoader(sets[0],batch_size=config.batch_size,shuffle=True,collate_fn=sets[0].collate_fn)
    if args.smoke:
        example=min((s for s in sets[0] if s.graph.roles), key=lambda s: len(s.tokens))
        loader=[sets[0].collate_fn([example])]
    updates=(len(loader)+config.accumulate_step-1)//config.accumulate_step
    schedule=get_linear_schedule_with_warmup(optimizer, updates*config.warmup_epoch, updates*config.max_epoch)
    best=-1.0
    start_epoch=0
    if args.resume and (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['schedule']);start_epoch=saved['epoch'];best=saved['best']
    started=time.time()
    max_epoch = 1 if args.smoke else args.epochs or config.max_epoch
    for epoch in range(start_epoch, max_epoch):
        model.train(); optimizer.zero_grad(); total=0.0
        for step,batch in enumerate(loader,1):
            loss=model(batch)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {epoch=} {step=}')
            (loss/config.accumulate_step).backward(); total+=loss.item()
            if step%config.accumulate_step==0 or step==len(loader) or args.smoke:
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clipping)
                optimizer.step();schedule.step();optimizer.zero_grad()
            if step%50==0 or args.smoke:
                print(json.dumps(dict(epoch=epoch+1,step=step,loss=total/step,seconds=time.time()-started)),flush=True)
            if args.smoke:
                with torch.no_grad(): prediction=model.predict(batch)
                assert len(prediction)==len(batch.graphs)
                (output/'smoke.json').write_text(json.dumps(dict(loss=loss.item(),loader_counts=[len(s) for s in sets],gpu=config.use_gpu)),encoding='utf-8')
                print('BATCH FORWARD/BACKWARD/DECODE PASS',flush=True)
                return
        model.eval(); gold=[];pred=[]
        with torch.no_grad():
            for batch in DataLoader(sets[1],batch_size=config.eval_batch_size,collate_fn=sets[1].collate_fn):
                graphs=model.predict(batch)
                for graph in graphs: graph.clean(relation_directional=False,symmetric_relations=[])
                gold.extend(batch.graphs);pred.extend(graphs)
        scores=score_graphs(gold,pred)
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(epoch=epoch+1,train_loss=total/len(loader),dev=scores))+'\n')
        if scores['role']['f']>best:
            best=scores['role']['f']
            torch.save(dict(model=model.state_dict(),config=config.to_dict(),vocabs=vocabs,valid=patterns),output/'best.role.mdl')
        torch.save(dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule.state_dict(),best=best),output/'last.pt')
    state=torch.load(output/'best.role.mdl',weights_only=False)
    model.load_state_dict(state['model']);model.eval();gold=[];pred=[]
    with torch.no_grad():
        for batch in DataLoader(sets[2],batch_size=config.eval_batch_size,collate_fn=sets[2].collate_fn):
            graphs=model.predict(batch)
            for graph in graphs: graph.clean(relation_directional=False,symmetric_relations=[])
            gold.extend(batch.graphs);pred.extend(graphs)
    (output/'test.json').write_text(json.dumps(score_graphs(gold,pred),indent=2),encoding='utf-8')
    print('TRAIN COMPLETE',flush=True)


if __name__=='__main__':main()
