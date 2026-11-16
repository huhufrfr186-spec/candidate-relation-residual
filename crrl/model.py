import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from .graph import local_graphs,grid_points,soft_transport,weighted_projection,BoundedCorrection


class FeatureBackbone(nn.Module):
    def __init__(self,channels,pretrained=False):
        super().__init__()
        r=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        for stage,dilation in [(r.layer3,2),(r.layer4,4)]:
            stage[0].conv1.stride=(1,1)
            stage[0].downsample[0].stride=(1,1)
            for block in stage:
                for conv in [block.conv1,block.conv2]:
                    conv.dilation=(dilation,dilation);conv.padding=(dilation,dilation)
        self.stages=nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool,r.layer1,r.layer2,r.layer3,r.layer4)
        self.projection=nn.Conv2d(512,channels,1)

    def forward(self,image):
        return self.projection(self.stages(image))


class CandidateRelationDetector(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg.copy();self.classes=cfg['classes'];d=cfg.get('channels',32)
        self.mode=cfg.get('output','probabilistic');self.context=cfg.get('context','residual')
        outputs=self.classes+1+(self.classes*10 if self.mode=='probabilistic' else 6)
        self.rgb=FeatureBackbone(d,cfg.get('pretrained',False));self.ir=FeatureBackbone(d,cfg.get('pretrained',False))
        self.edge=nn.Sequential(nn.Linear(3*d+2,64),nn.ReLU(),nn.Linear(64,d),nn.ReLU())
        self.compatibility=nn.Sequential(nn.Linear(3*d+2,64),nn.ReLU(),nn.Linear(64,1))
        self.direct=nn.Sequential(nn.Linear(3*d,64),nn.ReLU(),nn.Linear(64,outputs))
        self.rgb_head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,outputs))
        self.ir_head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,outputs))
        correction_input=3*d if self.context=='graph' else d
        self.correction=BoundedCorrection(correction_input,outputs,cfg.get('beta',.8),hidden=37 if self.context=='graph' else 64)
        self.unmatched=nn.Parameter(torch.tensor(0.))
        self._graphs={}
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('image_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        for head in [self.direct,self.rgb_head,self.ir_head]:
            nn.init.normal_(head[-1].weight,std=.01);nn.init.zeros_(head[-1].bias)
            with torch.no_grad():
                bias=head[-1].bias[self.classes+1:]
                if self.mode=='continuous':
                    bias[2:4]=-3.;bias[4]=1.
                else:
                    b=bias.view(self.classes,10);b[:,2:4]=-3.;b[:,4:8]=-3.;b[:,9]=-2.

    def anchor(self,raw,points):
        raw=raw.clone()
        if self.mode=='continuous':raw[...,self.classes+1:self.classes+3]+=points
        else:
            tail=raw[...,self.classes+1:].reshape(*raw.shape[:-1],self.classes,10)
            tail[...,:2]+=points.unsqueeze(-2)
            raw=torch.cat((raw[...,:self.classes+1],tail.flatten(-2)),-1)
        return raw

    def forward(self,rgb,ir,intervention=None):
        if rgb.shape!=ir.shape:raise ValueError('Paired image tensors must have equal shapes')
        r=self.rgb((rgb-self.image_mean)/self.image_std);t=self.ir((ir-self.image_mean)/self.image_std)
        grid=self.cfg.get('query_grid')
        if grid:
            r=F.interpolate(r,size=grid,mode='bilinear',align_corners=False)
            t=F.interpolate(t,size=grid,mode='bilinear',align_corners=False)
        batch,d,h,w=r.shape;r=r.flatten(2).transpose(1,2);t=t.flatten(2).transpose(1,2)
        key=(h,w,self.cfg.get('block',2),self.cfg.get('radius',2))
        if key not in self._graphs:self._graphs[key]=local_graphs(*key)
        points=grid_points(h,w,r.device,r.dtype)
        slots=(2*self.cfg.get('radius',2)+1)**2+1
        raw=r.new_zeros(batch,h*w,slots,self.direct[-1].out_features)
        mass=r.new_zeros(batch,h*w,slots)
        null=self.anchor(self.rgb_head(r),points)
        stage_errors=[]
        for g in self._graphs[key]:
            qi,ti=g.queries.to(r.device),g.candidates.to(r.device)
            edges=g.edges.to(r.device);er,ec=edges.T
            a,b=r[:,qi],t[:,ti]
            ea,eb=a[:,:,er,:],b[:,:,ec,:]
            relative=g.relative.to(r.device).unsqueeze(0).expand(batch,-1,-1,-1)
            pair=torch.cat((ea,eb,ea*eb,relative),-1)
            feature=self.edge(pair)
            score=r.new_zeros(batch,len(qi),qi.shape[1],ti.shape[1])
            score[:,:,er,ec]=self.compatibility(pair).squeeze(-1)
            plan=soft_transport(score,g.support,self.unmatched,self.cfg.get('temperature',.1),self.cfg.get('sinkhorn_iterations',200))
            p=plan[...,:-1,:-1][:,:,er,ec]
            if self.context=='none':z=torch.zeros_like(feature)
            elif self.context=='residual':
                z=weighted_projection(feature,p,g.design,self.cfg.get('eta',.05))
            elif self.context in ('graph','demean','equal_weight'):
                row=r.new_zeros(batch,len(qi),qi.shape[1],d)
                col=r.new_zeros(batch,len(qi),ti.shape[1],d)
                row.index_add_(2,er,feature);col.index_add_(2,ec,feature)
                nr=g.support.sum(-1).to(r.device).clamp_min(1)[None,None,:,None]
                nc=g.support.sum(0).to(r.device).clamp_min(1)[None,None,:,None]
                ra,cb=(row/nr)[:,:,er],(col/nc)[:,:,ec]
                if self.context=='graph':z=torch.cat((feature,ra,cb),-1)
                elif self.context=='demean':z=feature-ra-cb+feature.mean(-2,keepdim=True)
                else:z=weighted_projection(feature,torch.ones_like(p),g.design,self.cfg.get('eta',.05))
            else:raise ValueError(f'Unknown context: {self.context}')
            if intervention=='zero':z=torch.zeros_like(z)
            elif intervention=='shuffle':
                permutation=torch.randperm(z.shape[-2],device=z.device);z=z[:,:,permutation]
            if self.cfg.get('stop_transport_gradient',False) and self.context=='residual':
                z=weighted_projection(feature,p.detach(),g.design,self.cfg.get('eta',.05))
            value=self.anchor(self.direct(torch.cat((feature,ea,eb),-1)),points[qi[:,er]])
            if self.context!='none':value=value+self.correction(z)
            max_k=slots-1
            dense=r.new_zeros(batch,len(qi),qi.shape[1],max_k,raw.shape[-1])
            weights=r.new_zeros(batch,len(qi),qi.shape[1],max_k)
            for row_index in range(qi.shape[1]):
                edge_mask=er==row_index;k=int(edge_mask.sum())
                dense[:,:,row_index,:k]=value[:,:,edge_mask]
                weights[:,:,row_index,:k]=p[:,:,edge_mask].to(r.dtype)
            flat_q=qi.flatten()
            raw[:,flat_q,:max_k]=dense.flatten(1,2)
            raw[:,flat_q,max_k]=null[:,flat_q]
            mass[:,flat_q,:max_k]=weights.flatten(1,2)
            mass[:,flat_q,max_k]=plan[...,:-1,-1].flatten(1,2).to(r.dtype)
            row_error=(plan[...,:-1,:].sum(-1)-1).abs().max().detach()
            stage_errors.append(row_error)
        return {'raw':raw,'mass':mass,'rgb':null,'ir':self.anchor(self.ir_head(t),points),
                'points':points,'grid':(h,w),'output':self.mode,'classes':self.classes,
                'transport_error':torch.stack(stage_errors).max()}
