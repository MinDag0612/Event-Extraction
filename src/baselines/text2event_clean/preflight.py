"""Validate targets, teacher forcing, constraints and one optimizer update."""
import json
import hashlib
import random
import torch
from src.baselines.text2event.convert import parse, convert_substructure
from src.baselines.text2event.training import BundleDataset, BatchCollator, load_components
from src.baselines.text2event.upstream_decoding import UpstreamConstraint
from src.baselines.text2event.update_probe import UpdateProbe


def bundle_hashes(bundle):
    """Fingerprint TRAIN/DEV, schema and tokenizer artifacts for the audited run."""
    paths=[bundle/'GENEVA'/name for name in ['train.json','train.gold.jsonl','dev.json','dev.gold.jsonl','event.schema']]
    paths+=sorted(p for p in (bundle/'tokenizer').iterdir() if p.is_file())
    return {str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_batch(batch, tokenizer, model):
    """Validate padded labels and the exact decoder-input shift used by mT5."""
    labels=batch['labels']
    active=labels.ne(-100)
    if not active.any() or not torch.all(active[:,0]):raise ValueError('Invalid label mask')
    for row in labels:
        ids=row[row.ne(-100)].tolist()
        if ids.count(tokenizer.eos_token_id)!=1 or ids[-1]!=tokenizer.eos_token_id:
            raise ValueError('Target must terminate with exactly one EOS')
        if tokenizer.pad_token_id in ids:raise ValueError('Padding token included in target loss')
        padding=row.eq(-100).nonzero()
        if len(padding) and not torch.all(row[padding[0].item():].eq(-100)):
            raise ValueError('Non-trailing label padding')
    shifted=model.prepare_decoder_input_ids_from_labels(labels=labels)
    expected=torch.full_like(labels,tokenizer.pad_token_id)
    expected[:,0]=model.config.decoder_start_token_id
    expected[:,1:]=labels[:,:-1].masked_fill(labels[:,:-1].eq(-100),tokenizer.pad_token_id)
    if not torch.equal(shifted,expected):raise ValueError('Decoder input shift mismatch')
    return shifted


def audit_data(bundle, tokenizer, config, output):
    """Validate every TRAIN/DEV target and fixed gold prefixes for both stages."""
    folder=bundle/'GENEVA';report={};datasets={}
    normalize=lambda text:' '.join(tokenizer.backend_tokenizer.normalizer.normalize_str(text).split())
    for split in ['train','dev']:
        gold=list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines()))
        selected=sorted(random.Random(421).sample(range(len(gold)),8))
        for stage in ['substructure','full']:
            prefix=config[f'{stage}_source_prefix']
            ds=BundleDataset(folder,split,tokenizer,config['max_source_length'],config['max_target_length'],stage,prefix)
            for index,sample in enumerate(gold):
                target=convert_substructure(sample)['event'] if stage=='substructure' else None
                if target is None:
                    from src.baselines.text2event.convert import convert
                    target=convert(sample)['event']
                ids=ds[index]['labels']
                decoded=tokenizer.decode(ids[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if normalize(json.dumps(parse(target),ensure_ascii=False))!=normalize(json.dumps(parse(decoded),ensure_ascii=False)):
                    raise ValueError(f'Target roundtrip mismatch: {sample["id"]}/{stage}')
            for index in selected:
                item=ds[index];prefix_length=len(tokenizer.encode(prefix,add_special_tokens=False))
                constraint=UpstreamConstraint(tokenizer,folder/'event.schema',item['input_ids'][prefix_length:],flat=stage=='substructure')
                sequence=[tokenizer.pad_token_id]
                for position,token in enumerate(item['labels']):
                    if token not in constraint(0,torch.tensor(sequence)):
                        raise ValueError(f'Gold token masked: {item["id"]}/{stage}/{position}')
                    sequence.append(token)
            report[f'{split}/{stage}']={'rows':len(ds),'roundtrip_rows':len(ds),
                'gold_prefix_ids':[gold[i]['id'] for i in selected],
                'max_source_tokens':max(len(r['input_ids']) for r in ds),
                'max_target_tokens':max(len(r['labels']) for r in ds)}
            datasets[split,stage]=ds
    (output/'data-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return datasets


def run(bundle, config, recipe, output):
    """Load fresh pretrained weights, audit data, then measure one diagnostic update."""
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    def event(name):print(json.dumps({'phase':name}),flush=True)
    random.seed(recipe['seed']);torch.manual_seed(recipe['seed'])
    save('request.json',{'dataset':'GENEVA','initialization':'fresh-pinned-pretrained',
        'diagnostic_optimizer_steps':1,'baseline_training_started':False,'test_used':False,
        'config':config,'recipe':recipe,'bundle_hashes':bundle_hashes(bundle)})
    event('MODEL_LOAD')
    model,t=load_components(config,bundle/'tokenizer')
    save('model-loading.json',model.text2event_load_report)
    event('DATA_AUDIT')
    datasets=audit_data(bundle,t,config,output)
    model.to('cuda');model.gradient_checkpointing_enable();model.config.use_cache=False
    ds=datasets['train','substructure']
    selected=sorted(range(len(ds)),key=lambda i:len(ds[i]['labels']))
    indices=[selected[0],selected[-1]]
    batch=BatchCollator(t)([ds[i] for i in indices])
    values={k:v.to('cuda') for k,v in batch.items() if k!='ids'}
    decoder_ids=validate_batch(values,t,model)
    event('SHIFT_AND_LOSS_AUDIT')
    model.eval()
    with torch.no_grad():
        automatic=model(**values,use_cache=False)
        explicit=model(**values,decoder_input_ids=decoder_ids,use_cache=False)
        shift_equal=torch.allclose(automatic.logits,explicit.logits,atol=1e-5,rtol=1e-5)
        manual=torch.nn.functional.cross_entropy(automatic.logits.flatten(0,1),values['labels'].flatten(),ignore_index=-100)
        if not shift_equal or not torch.allclose(automatic.loss,manual,atol=1e-5,rtol=1e-5):
            raise ValueError('Teacher forcing loss/shift mismatch')
        before_loss=automatic.loss.item()
    del automatic,explicit,manual
    event('OPTIMIZER_AUDIT')
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    loss=model(**values,use_cache=False).loss
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    probe=UpdateProbe(output/'optimizer-step.jsonl',[t.convert_tokens_to_ids(x) for x in config['additional_special_tokens']])
    probe.stage='preflight';probe.epoch=0
    probe.before_step(model,optimizer,norm);optimizer.step();probe.after_step(model)
    record=json.loads((output/'optimizer-step.jsonl').read_text())
    if norm<=0 or any(v['max_abs']<=0 for v in record['parameter_deltas'].values()):
        raise ValueError('Observed parameters were not updated')
    optimizer.zero_grad(set_to_none=True);model.eval()
    with torch.no_grad():after_loss=model(**values,use_cache=False).loss.item()
    save('completion.json',{'status':'passed','batch_ids':batch['ids'],'loss_before_update':before_loss,
        'loss_after_update':after_loss,'manual_loss_matches':True,'decoder_shift_matches':True,
        'observed_parameter_updates':record,'baseline_checkpoint_saved':False,'test_used':False})
    event('PREFLIGHT_COMPLETE')
