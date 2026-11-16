import torch
from torch import nn
from torch.nn import functional as F
from .assignment import assign_targets
from .geometry import encode_boxes
from .probability import bin_widths,box_region,mixture_region_log_prob


class DetectionLoss(nn.Module):
    def __init__(self,model_cfg,loss_cfg):
        super().__init__();self.model_cfg=model_cfg;self.cfg=loss_cfg

    def branch(self,raw,mass,owner,target,image_size,classes,mode):
        positive=owner>=0
        labels=torch.full_like(owner,classes)
        labels[positive]=target['labels'][owner[positive]]
        if mode=='continuous':
            combined=(mass.unsqueeze(-1)*raw).sum(-2)
            cls=F.cross_entropy(combined[:,:classes+1],labels)
            reg=combined.sum()*0
            if positive.any():
                code=encode_boxes(target['boxes'][owner[positive]],image_size)
                reg=F.smooth_l1_loss(combined[positive,classes+1:],code,reduction='sum')/positive.sum()
            return cls+self.cfg.get('box_weight',1.)*reg
        widths=bin_widths(image_size,**self.cfg.get('bins',{}))
        limits=self.cfg.get('distribution',{})
        log_cls=raw[...,:classes+1].double().log_softmax(-1)
        lw=mass.double().clamp_min(1e-300).log().masked_fill(mass<=0,-torch.inf)
        negative=~positive
        # Both sets are divided by the number of valid queries.
        value=raw.sum()*0
        if negative.any():value=value-torch.logsumexp(lw[negative]+log_cls[negative,:,classes],-1).sum()
        if positive.any():
            regions=box_region(target['boxes'][owner[positive]],widths,image_size)
            value=value-mixture_region_log_prob(raw[positive],mass[positive],labels[positive],regions,classes,widths,limits).sum()
        return value/len(owner)

    def forward(self,prediction,targets,image_size):
        losses=[];main=[];aux=[]
        for b,target in enumerate(targets):
            owner=assign_targets(prediction['points'],target['boxes'],image_size,
                                 self.cfg.get('center_fraction',.5),self.cfg.get('assignment','unique'))
            c,mode=prediction['classes'],prediction['output']
            fused=self.branch(prediction['raw'][b],prediction['mass'][b],owner,target,image_size,c,mode)
            unit=prediction['mass'].new_ones(len(owner),1)
            rgb=self.branch(prediction['rgb'][b].unsqueeze(-2),unit,owner,target,image_size,c,mode)
            infrared={'boxes':target['ir_boxes'],'labels':target['ir_labels']}
            ir_owner=assign_targets(prediction['points'],infrared['boxes'],image_size,
                                    self.cfg.get('center_fraction',.5),self.cfg.get('assignment','unique'))
            ir=self.branch(prediction['ir'][b].unsqueeze(-2),unit,ir_owner,infrared,image_size,c,mode)
            extra=self.cfg.get('rgb_aux_weight',1.)*rgb+self.cfg.get('ir_aux_weight',1.)*ir
            main.append(fused);aux.append(extra);losses.append(fused+extra)
        return {'loss':torch.stack(losses).mean(),'main':torch.stack(main).mean(),
                'auxiliary':torch.stack(aux).mean(),'transport_error':prediction['transport_error']}
