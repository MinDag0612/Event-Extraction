"""Bounded pilot selection and disk-aware weights-only checkpoint utilities."""
import random
import shutil
from pathlib import Path
import torch


def fixed_subset(rows, size, seed=421):
    if size<1 or size>len(rows):raise ValueError('Invalid subset size')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
    return random.Random(seed).sample(rows,size)


def weights_bytes(state):
    seen=set();total=0
    pending=[state]
    while pending:
        tensor=pending.pop()
        if isinstance(tensor,dict):pending.extend(tensor.values());continue
        if isinstance(tensor,(tuple,list)):pending.extend(tensor);continue
        if not isinstance(tensor,torch.Tensor):continue
        storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
        if key not in seen:
            seen.add(key);total+=storage.nbytes()
    return total


def save_state_atomic(path,state):
    """Disk preflight includes optimizer tensors; preserve prior completed file."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    estimated=weights_bytes(state);free=shutil.disk_usage(path.parent).free
    required=int(estimated*1.05)+256*1024**2
    if free<required:
        raise OSError(f'Insufficient free disk for atomic checkpoint: free={free}, required={required}, path={path}')
    temporary=path.with_suffix('.tmp')
    try:
        torch.save(state,temporary);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    return {'estimated_tensor_bytes':estimated,'disk_free_before':free,'disk_free_after':shutil.disk_usage(path.parent).free}


def save_weights_atomic(path, model, **metadata):
    """Keep previous completed file on failure; refuse before filling the disk.

    Only weights and metadata, not exact optimizer/RNG/sampler resume state.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=model.state_dict();estimated=weights_bytes(state)
    free=shutil.disk_usage(path.parent).free
    required=int(estimated*1.05)+256*1024**2
    if free<required:
        raise OSError(f'Insufficient free disk for atomic weights: free={free}, required={required}, path={path}')
    temporary=path.with_suffix('.tmp')
    try:
        torch.save({'model':state,'kind':'weights-only-pilot',**metadata},temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'estimated_weights_bytes':estimated,'disk_free_before':free,
            'disk_free_after':shutil.disk_usage(path.parent).free}
