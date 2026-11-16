import math
import numpy as np
import torch
from torch.nn import functional as F


def log_normal_interval(lower, upper):
    lower,upper=torch.broadcast_tensors(lower,upper)
    result=torch.empty_like(lower)
    for mask,tail in ((lower>=0,True),(lower<0,False)):
        if not mask.any():continue
        a,b=lower[mask],upper[mask]
        high=torch.special.log_ndtr(-a if tail else b)
        low=torch.special.log_ndtr(-b if tail else a)
        ratio=(low-high).clamp_max(-torch.finfo(high.dtype).eps)
        result[mask]=high+torch.log(-torch.expm1(ratio))
    return result


def angular_log_mass(lower, upper, mean, kappa, order=24):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    nodes=mean.new_tensor(nodes);weights=mean.new_tensor(weights)
    lo,hi,mu,k=torch.broadcast_tensors(lower,upper,mean,kappa)
    half=(hi-lo)/2
    angle=lo.unsqueeze(-1)+(nodes+1)*half.unsqueeze(-1)
    log_norm=math.log(math.pi)+torch.log(torch.special.i0e(k))+k
    log_density=k.unsqueeze(-1)*torch.cos(2*(angle-mu.unsqueeze(-1)))-log_norm.unsqueeze(-1)
    return torch.logsumexp(log_density+weights.log()+half.log().unsqueeze(-1),-1)


def distribution_parameters(raw,classes,sigma_min=1e-4,sigma_max=.5,kappa_max=100.):
    log_class=F.log_softmax(raw[...,:classes+1].double(),-1)
    box=raw[...,classes+1:].double().reshape(*raw.shape[:-1],classes,10)
    return log_class,box[...,:4],sigma_min+(sigma_max-sigma_min)*box[...,4:8].sigmoid(),box[...,8]%math.pi,kappa_max*box[...,9].sigmoid()


def bin_widths(image_size,center_pixels=2.,log_scale=.05,angle_degrees=2.):
    width,height=image_size
    angle_bins=round(180/angle_degrees)
    if abs(angle_bins*angle_degrees-180)>1e-6:
        raise ValueError("angle_degrees must partition 180 degrees")
    return [center_pixels/width,center_pixels/height,log_scale,log_scale,math.pi/angle_bins]


def box_region(boxes,widths,image_size):
    width,height=image_size
    value=torch.cat((boxes[...,:2]/boxes.new_tensor([width,height]),
                     (boxes[...,2:4]/boxes.new_tensor([width,height])).clamp_min(1e-10).log(),
                     (boxes[...,4:]%math.pi)), -1)
    steps=value.new_tensor(widths)
    index=torch.floor(value/steps).long()
    index[...,-1]%=round(math.pi/widths[-1])
    return index


def component_region_log_prob(parameters, region, widths):
    mu,sigma,angle,kappa=parameters
    region=region.to(mu.device)
    step=mu.new_tensor(widths)
    lower=region*step;upper=lower+step
    normal=log_normal_interval((lower[...,:4]-mu)/sigma,(upper[...,:4]-mu)/sigma).sum(-1)
    return normal+angular_log_mass(lower[...,4],upper[...,4],angle,kappa)


def mixture_region_log_prob(raw,mass,labels,regions,classes,widths,limits):
    log_class,mu,sigma,angle,kappa=distribution_parameters(raw,classes,**limits)
    index=torch.arange(len(raw),device=raw.device)
    c=labels.long()
    # raw: query x component x parameters; one supervised event per query.
    pars=(mu[index,:,c],sigma[index,:,c],angle[index,:,c],kappa[index,:,c])
    log_box=component_region_log_prob(pars,regions[:,None,:],widths)
    log_w=mass.double().clamp_min(1e-300).log().masked_fill(mass<=0,-torch.inf)
    return torch.logsumexp(log_w+log_class[index,:,c]+log_box,-1)
