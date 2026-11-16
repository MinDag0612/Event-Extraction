"""Full-DEV stopping policy for the epoch-sized PHEE controller."""
import math

POLICY = {'enabled': True, 'metric': 'full DEV949 AC F1', 'patience': 3,
          'min_delta': 0.001, 'min_full_epochs': 5, 'max_full_epochs': 30}


def stopping_state(evaluations, policy):
    """Replay consecutive complete full-DEV scores; keep patience separate from best."""
    for key,value in POLICY.items():
        if policy.get(key)!=value: raise ValueError('Unexpected early stopping policy: '+key)
    ordered=sorted(evaluations,key=lambda row:row['epoch'])
    if not ordered or ordered[0]['epoch']!=3:
        raise ValueError('Full-DEV history must start at epoch3')
    reference=None;bad=0;best=-1.;best_epoch=None;last=2;stopped=False
    for row in ordered:
        epoch=row['epoch'];score=row['score']
        if epoch!=last+1 or not row.get('dev_complete') or row.get('dev_samples')!=949:
            raise ValueError('Require consecutive complete DEV949 evaluations')
        if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid AC score')
        if stopped:raise ValueError('History continues after an early-stop decision')
        if score>best:best=score;best_epoch=epoch
        if reference is None or score>reference+policy['min_delta']:
            reference=score;bad=0
        elif epoch>policy['min_full_epochs']:
            bad+=1
        stopped=bad>=policy['patience'];last=epoch
    return {**POLICY,'reference':reference,'bad_epochs':bad,'epochs_seen':last,'stop_epoch':last if stopped else None,
            'stopped':stopped,'best_score':best,'best_epoch':best_epoch}


def permits_final_evaluation(epoch, early_stopping):
    """Never open TEST early without a persisted matching stopping decision."""
    if epoch==30:return True
    return bool(early_stopping and all(early_stopping.get(k)==v for k,v in POLICY.items())
                and early_stopping.get('stopped') is True
                and early_stopping.get('stop_epoch')==epoch
                and early_stopping.get('epochs_seen')==epoch
                and epoch>=8 and early_stopping.get('bad_epochs',0)>=3)
