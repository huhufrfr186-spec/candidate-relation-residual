import itertools
import math
import numpy as np
import torch
from .geometry import decode_boxes,rotated_nms
from .probability import distribution_parameters,bin_widths,log_normal_interval,angular_log_mass


def region_candidates(mean,angle,kappa,widths,uniform_tolerance=1e-4):
    mean=mean.detach().cpu().numpy();angle=angle.detach().cpu().numpy();kappa=kappa.detach().cpu().numpy()
    steps=np.asarray(widths);bins=round(math.pi/widths[-1]);regions=set()
    for center,phi,concentration in zip(mean,angle,kappa):
        normal=np.floor(center/steps[:4]).astype(np.int64)
        axes=[[int(v-1),int(v),int(v+1)] for v in normal]
        ab=int(math.floor((float(phi)%math.pi)/widths[-1]))%bins
        axes.append(list(range(bins)) if concentration<=uniform_tolerance else sorted({(ab-1)%bins,ab,(ab+1)%bins}))
        regions.update(itertools.product(*axes))
    return np.asarray(sorted(regions),dtype=np.int64).reshape(-1,5)


def score_regions(parameters,component_log_weight,regions,widths,chunk_size=2048):
    mean,sigma,angle,kappa=parameters
    device=mean.device;ids=torch.as_tensor(regions,device=device)
    caches=[];inverse=[]
    for dimension in range(5):
        unique,inv=torch.unique(ids[:,dimension],sorted=True,return_inverse=True)
        lower=unique.to(mean.dtype)*widths[dimension];upper=lower+widths[dimension]
        if dimension<4:
            logp=log_normal_interval((lower[:,None]-mean[None,:,dimension])/sigma[None,:,dimension],
                                    (upper[:,None]-mean[None,:,dimension])/sigma[None,:,dimension])
        else:
            logp=angular_log_mass(lower[:,None],upper[:,None],angle[None,:],kappa[None,:])
        caches.append(logp);inverse.append(inv)
    result=[]
    for start in range(0,len(ids),chunk_size):
        terms=component_log_weight.unsqueeze(0).expand(min(chunk_size,len(ids)-start),-1)
        for cache,inv in zip(caches,inverse):terms=terms+cache[inv[start:start+chunk_size]]
        result.append(torch.logsumexp(terms,-1))
    return torch.cat(result) if result else mean.new_empty(0)


def refine_representatives(regions,parameters,component_log_weight,widths,steps=20):
    mean,sigma,angle,kappa=[p.detach() for p in parameters]
    regions=torch.as_tensor(regions,device=mean.device,dtype=mean.dtype)
    lo=regions*mean.new_tensor(widths);hi=lo+mean.new_tensor(widths)
    coordinate=torch.zeros_like(lo,requires_grad=True)
    with torch.enable_grad():
        optimizer=torch.optim.Adam([coordinate],lr=.1)
        for _ in range(steps):
            optimizer.zero_grad()
            x=lo+(hi-lo)*coordinate.sigmoid()
            delta=(x[:,None,:4]-mean[None])/sigma[None]
            normal=-.5*delta.square().sum(-1)-sigma.log().sum(-1)[None]-2*math.log(2*math.pi)
            angular=kappa[None]*torch.cos(2*(x[:,None,4]-angle[None]))-math.log(math.pi)-torch.log(torch.special.i0e(kappa))[None]-kappa[None]
            value=torch.logsumexp(component_log_weight[None]+normal+angular,-1).mean()
            (-value).backward()
            optimizer.step()
    return (lo+(hi-lo)*coordinate.sigmoid()).detach()


class Decoder:
    def __init__(self,model_cfg,loss_cfg,decode_cfg):
        self.model_cfg=model_cfg;self.loss_cfg=loss_cfg;self.cfg=decode_cfg

    @torch.no_grad()
    def __call__(self,prediction,image_size):
        all_detections=[];classes=prediction['classes']
        for b in range(len(prediction['raw'])):
            raw,mass=prediction['raw'][b],prediction['mass'][b]
            if prediction['output']=='continuous':
                value=(mass.unsqueeze(-1)*raw).sum(-2)
                scores=value[:,:classes+1].softmax(-1)[:,:classes]
                code=value[:,classes+1:].clone()
                degenerate=code[:,4:6].square().sum(-1)<1e-16
                if degenerate.any():
                    angular=raw[...,classes+5:classes+7]
                    valid=angular.square().sum(-1)>1e-16
                    ranked=mass.masked_fill(~valid,-1).argmax(-1)
                    replacement=angular[torch.arange(len(raw),device=raw.device),ranked]
                    code[degenerate,4:6]=replacement[degenerate]
                boxes=decode_boxes(code,image_size)
                q,c=(scores>=self.cfg.get('score_threshold',.001)).nonzero(as_tuple=True)
                all_boxes=boxes[q].cpu().numpy();all_scores=scores[q,c].cpu().numpy();all_labels=c.cpu().numpy()
                queries=q.cpu().numpy()
            else:
                widths=bin_widths(image_size,**self.loss_cfg.get('bins',{}))
                limits=self.loss_cfg.get('distribution',{})
                logc,mu,sigma,angle,kappa=distribution_parameters(raw,classes,**limits)
                all_boxes=[];all_scores=[];all_labels=[];queries=[]
                for q in range(len(raw)):
                    valid=mass[q]>0
                    for c in range(classes):
                        pars=(mu[q,valid,c],sigma[q,valid,c],angle[q,valid,c],kappa[q,valid,c])
                        logweight=mass[q,valid].double().log()+logc[q,valid,c]
                        regions=region_candidates(pars[0],pars[2],pars[3],widths,self.cfg.get('uniform_tolerance',1e-4))
                        logs=score_regions(pars,logweight,regions,widths,self.cfg.get('region_chunk_size',2048))
                        keep=logs>=math.log(max(self.cfg.get('score_threshold',.001),1e-300))
                        if not keep.any():continue
                        selected=regions[keep.cpu().numpy()]
                        if self.cfg.get('refine',False):
                            values=refine_representatives(selected,pars,logweight,widths,self.cfg.get('refinement_steps',20))
                        else:values=(torch.as_tensor(selected,device=raw.device,dtype=torch.float64)+.5)*raw.new_tensor(widths)
                        width,height=image_size
                        boxes=torch.cat((values[:,:2]*values.new_tensor([width,height]),
                                         values[:,2:4].clamp(-12,4).exp()*values.new_tensor([width,height]),
                                         (values[:,4:]%math.pi)),-1).cpu().numpy()
                        all_boxes.extend(boxes.tolist());all_scores.extend(logs[keep].exp().cpu().tolist())
                        all_labels.extend([c]*len(boxes));queries.extend([q]*len(boxes))
                all_boxes=np.asarray(all_boxes,dtype=float).reshape(-1,5)
                all_scores=np.asarray(all_scores);all_labels=np.asarray(all_labels,dtype=int);queries=np.asarray(queries,dtype=int)
            keep=rotated_nms(all_boxes,all_scores,all_labels,self.cfg.get('nms_iou',.5),self.cfg.get('max_detections',300))
            all_detections.append({'boxes':all_boxes[keep].tolist(),'scores':all_scores[keep].tolist(),
                                   'labels':all_labels[keep].tolist(),'query_ids':queries[keep].tolist()})
        return all_detections
