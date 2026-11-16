"""Read-only optimizer telemetry. No forward pass, random sampling or training decisions."""
import json
import torch


class UpdateProbe:
    def __init__(self,path,marker_ids):
        self.path=path;self.marker_ids=marker_ids;self.step=0
        self.stage=None;self.epoch=None

    def snapshots(self,model):
        result={}
        for name,module in [('input_markers',model.get_input_embeddings()),('output_markers',model.get_output_embeddings())]:
            result[name]=module.weight.detach()[self.marker_ids].float().clone()
        for name,p in model.named_parameters():
            if name.startswith('decoder.') and p.ndim==2 and p.requires_grad:
                result['decoder_sample:'+name]=p.detach().flatten()[:64].float().clone();break
        return result

    def before_step(self,model,optimizer,gradient_norm):
        self.before=self.snapshots(model)
        self.pending={'stage':self.stage,'epoch':self.epoch,'step':self.step+1,
                      'lr_before_step':[g['lr'] for g in optimizer.param_groups],
                      'gradient_norm_before_clip':float(gradient_norm)}

    def after_step(self,model):
        after=self.snapshots(model);deltas={}
        for key,before in self.before.items():
            delta=after[key]-before
            deltas[key]={'max_abs':float(delta.abs().max()),'l2':float(delta.norm())}
        self.pending['parameter_deltas']=deltas;self.step+=1
        with self.path.open('a') as stream:stream.write(json.dumps(self.pending)+'\n')
        del self.before
