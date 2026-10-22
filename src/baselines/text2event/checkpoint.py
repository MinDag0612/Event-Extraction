"""Atomic full-state checkpoints at epoch boundaries, for trusted local files only."""
import os
import random
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, signature, early_stopping=None):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'epoch':epoch,'best':best,'signature':signature,'python_rng':random.getstate(),
           'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if early_stopping is not None: state['early_stopping']=early_stopping.state_dict()
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);os.replace(temporary,path)


def restore_checkpoint(path,model,optimizer,scheduler,signature,early_stopping=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['signature']!=signature: raise ValueError('Resume configuration or data changed')
    if early_stopping is not None:
        if 'early_stopping' not in state: raise ValueError('Checkpoint lacks early stopping state')
        early_stopping.load_state_dict(state['early_stopping'])
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available(): raise ValueError('CUDA resume requires CUDA')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['epoch'],state['best']


def save_best(path, model, epoch, best, signature):
    """Evaluation checkpoint; optimizer/RNG live only in last.pt."""
    temporary=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'epoch':epoch,'best':best,'signature':signature},temporary)
    os.replace(temporary,path)
