import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import save_config
from .data import PairedDataset,collate,move_targets,invert_boxes
from .model import CandidateRelationDetector
from .losses import DetectionLoss
from .decode import Decoder
from .metrics import evaluate_detection


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def resolve_device(name):
    if name=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA device requested but not available')
    return device


def loader_for(cfg,split,augment=False,seed=17):
    dataset=PairedDataset(cfg['data'][split],cfg['data']['image_size'],augment,seed)
    if len(dataset)==0:raise ValueError(f'The {split} dataset is empty')
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=cfg['train'].get('batch_size',1),shuffle=augment,
                      num_workers=cfg['data'].get('workers',0),collate_fn=collate,generator=generator,
                      pin_memory=cfg.get('device','auto')!='cpu')
    return loader


def ensure_capacity(cfg,datasets):
    width,height=cfg['data']['image_size'];grid=cfg['model'].get('query_grid') or [math.ceil(height/8),math.ceil(width/8)]
    largest=max((max(len(s.get('annotations',[])),len(s.get('ir_annotations',[]))) for ds in datasets for s in ds.samples),default=0)
    if largest>grid[0]*grid[1]:
        factor=math.ceil(math.sqrt(largest/(grid[0]*grid[1])))
        grid=[grid[0]*factor,grid[1]*factor]
    cfg['model']['query_grid']=grid


def check_splits(train,val):
    train_ids={s['id'] for s in train.samples};val_ids={s['id'] for s in val.samples}
    train_images={str(Path(s['rgb']).resolve()) for s in train.samples}
    val_images={str(Path(s['rgb']).resolve()) for s in val.samples}
    if train_ids & val_ids or train_images & val_images:raise ValueError('Training and validation sets overlap')


def atomic_checkpoint(path,payload):
    temp=Path(str(path)+'.tmp');torch.save(payload,temp);os.replace(temp,path)


def load_checkpoint(path,device):
    state=torch.load(path,map_location=device,weights_only=False)
    cfg=copy.deepcopy(state['config']);cfg['model']['pretrained']=False
    model=CandidateRelationDetector(cfg['model']).to(device);model.load_state_dict(state['model'])
    return model,cfg,state


@torch.no_grad()
def predict_loader(model,loader,cfg,device,intervention=None):
    model.eval();decoder=Decoder(cfg['model'],cfg['loss'],cfg['decode']);rows=[]
    for rgb,ir,targets in loader:
        output=model(rgb.to(device),ir.to(device),intervention)
        predictions=decoder(output,tuple(cfg['data']['image_size']))
        for prediction,target in zip(predictions,targets):
            prediction['boxes']=invert_boxes(prediction['boxes'],target['transform']).tolist()
            prediction['id']=target['id'];rows.append(prediction)
    return rows


def train(cfg,output,resume=None):
    cfg=copy.deepcopy(cfg);seed=cfg.get('seed',17);seed_all(seed);torch.set_num_threads(cfg.get('cpu_threads',4))
    output=Path(output);output.mkdir(parents=True,exist_ok=True);device=resolve_device(cfg.get('device','auto'))
    train_loader=loader_for(cfg,'train',True,seed);val_loader=loader_for(cfg,'val',False,seed) if cfg['data'].get('val') else None
    if val_loader:check_splits(train_loader.dataset,val_loader.dataset)
    ensure_capacity(cfg,[train_loader.dataset]+([val_loader.dataset] if val_loader else []))
    if len(train_loader.dataset.classes)!=cfg['model']['classes']:raise ValueError('Dataset classes and model classes disagree')
    model=CandidateRelationDetector(cfg['model']).to(device)
    backbone_ids={id(p) for module in [model.rgb,model.ir] for p in module.parameters()}
    backbone=[p for p in model.parameters() if id(p) in backbone_ids];heads=[p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':cfg['train']['backbone_lr']},
                                 {'params':heads,'lr':cfg['train']['head_lr']}],weight_decay=cfg['train'].get('weight_decay',1e-4))
    criterion=DetectionLoss(cfg['model'],cfg['loss']);start=0;best=-float('inf');step=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['config']['model']!={**cfg['model']}:raise ValueError('Resume model configuration differs from the checkpoint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['epoch']+1;best=state['best'];step=state['step']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda' and state.get('cuda_rng'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        train_loader.generator.set_state(state['loader_rng'].cpu())
    save_config(cfg,output/'config.yaml')
    for epoch in range(start,cfg['train']['epochs']):
        model.train();train_loader.dataset.epoch=epoch;optimizer.zero_grad(set_to_none=True)
        accumulated={};began=time.perf_counter();accumulation=cfg['train'].get('accumulation',8)
        for index,(rgb,ir,targets) in enumerate(train_loader):
            prediction=model(rgb.to(device),ir.to(device))
            loss=criterion(prediction,move_targets(targets,device),tuple(cfg['data']['image_size']))
            if not torch.isfinite(loss['loss']):raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {index}')
            group_start=index//accumulation*accumulation;divisor=min(accumulation,len(train_loader)-group_start)
            (loss['loss']/divisor).backward()
            if (index+1)%accumulation==0 or index+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train'].get('gradient_clip',10.))
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
            for key,value in loss.items():accumulated[key]=accumulated.get(key,0.)+float(value.detach())
        record={'epoch':epoch+1,'step':step,'seconds':time.perf_counter()-began,
                **{key:value/len(train_loader) for key,value in accumulated.items()}}
        improved=False
        if val_loader and (epoch+1)%cfg['train'].get('evaluation_interval',1)==0:
            predictions=predict_loader(model,val_loader,cfg,device)
            metrics=evaluate_detection(predictions,val_loader.dataset.samples,val_loader.dataset.classes)
            record['validation']=metrics
            if metrics['AP50:95']>best:best=metrics['AP50:95'];improved=True
            (output/'validation_predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'epoch':epoch,'step':step,'best':best,
                 'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'loader_rng':train_loader.generator.get_state()}
        atomic_checkpoint(output/'last.pt',payload)
        if improved:atomic_checkpoint(output/'best.pt',payload)
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    return output/'last.pt'


def evaluate(checkpoint,manifest,output,device='auto',intervention=None):
    device=resolve_device(device);model,cfg,_=load_checkpoint(checkpoint,device)
    torch.set_num_threads(cfg.get('cpu_threads',4));cfg['data']['test']=str(manifest)
    loader=loader_for(cfg,'test');rows=predict_loader(model,loader,cfg,device,intervention)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics=evaluate_detection(rows,loader.dataset.samples,loader.dataset.classes)
    (output/'predictions.json').write_text(json.dumps(rows),encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics));return metrics


def benchmark(checkpoint,manifest,output,warmup=100,iterations=500,device='auto'):
    device=resolve_device(device);model,cfg,_=load_checkpoint(checkpoint,device);model.eval()
    cfg['data']['test']=str(manifest);loader=loader_for(cfg,'test');decoder=Decoder(cfg['model'],cfg['loss'],cfg['decode'])
    torch.set_num_threads(cfg.get('cpu_threads',4));rgb,ir,_=next(iter(loader));rgb,ir=rgb[:1].to(device),ir[:1].to(device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    rows=[]
    with torch.no_grad():
        for index in range(warmup+iterations):
            sync();t0=time.perf_counter();prediction=model(rgb,ir);sync();t1=time.perf_counter()
            decoder(prediction,tuple(cfg['data']['image_size']));sync();t2=time.perf_counter()
            if index>=warmup:rows.append({'model_ms':(t1-t0)*1000,'decoding_nms_ms':(t2-t1)*1000,'total_ms':(t2-t0)*1000})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for name in rows[0]:
        values=np.array([r[name] for r in rows]);summary[name]={'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(values)>1 else 0.}
    summary['frames_per_second']=1000/summary['total_ms']['mean']
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
