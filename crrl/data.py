import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance
from torch.utils.data import Dataset


def read_manifest(path):
    path=Path(path).resolve()
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or 'samples' not in data or 'classes' not in data:
        raise ValueError('A manifest must contain classes and samples')
    for sample in data['samples']:
        for field in ['rgb','ir']:
            p=Path(sample[field]);sample[field]=str(p if p.is_absolute() else path.parent/p)
    return data


class PairedDataset(Dataset):
    def __init__(self,manifest,image_size=(640,512),augment=False,seed=17):
        self.manifest=read_manifest(manifest);self.samples=self.manifest['samples']
        self.classes=self.manifest['classes'];self.image_size=tuple(image_size)
        self.augment=augment;self.seed=seed;self.epoch=0

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        sample=self.samples[index]
        rgb=Image.open(sample['rgb']).convert('RGB');ir=Image.open(sample['ir']).convert('RGB')
        rng=np.random.default_rng(self.seed+self.epoch*1000003+index)
        width,height=self.image_size
        def prepare(image,annotations):
            original=image.size
            # Uniform letterboxing preserves oriented rectangles and angles.
            scale=min(width/original[0],height/original[1]);nw,nh=round(original[0]*scale),round(original[1]*scale)
            dx,dy=(width-nw)//2,(height-nh)//2
            resized=image.resize((nw,nh),Image.Resampling.BILINEAR)
            canvas=Image.new('RGB',(width,height),(114,114,114));canvas.paste(resized,(dx,dy))
            boxes=np.asarray([a['bbox'] for a in annotations],dtype=np.float32).reshape(-1,5)
            if len(boxes):
                boxes[:,:4]*=scale;boxes[:,0]+=dx;boxes[:,1]+=dy
            labels=np.asarray([a['category_id'] for a in annotations],dtype=np.int64)
            if len(labels) and ((labels<0).any() or (labels>=len(self.classes)).any()):raise ValueError('Category IDs must be zero-based')
            return canvas,boxes,labels,{'scale':scale,'offset':[dx,dy],'size':list(original)}
        rgb,boxes,labels,transform=prepare(rgb,sample.get('annotations',[]))
        ir,ir_boxes,ir_labels,ir_transform=prepare(ir,sample.get('ir_annotations',[]))
        if self.augment and rng.random()<.5:
            rgb,ir=ImageOps.mirror(rgb),ImageOps.mirror(ir)
            for b in [boxes,ir_boxes]:
                b[:,0]=width-b[:,0];b[:,4]=(-b[:,4])%np.pi
        if self.augment:
            rgb=ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(.8,1.2)))
        tensor=lambda image:torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        target={'boxes':torch.from_numpy(boxes),'labels':torch.from_numpy(labels),
                'ir_boxes':torch.from_numpy(ir_boxes),'ir_labels':torch.from_numpy(ir_labels),
                'id':str(sample['id']),'transform':transform,'ir_transform':ir_transform}
        return tensor(rgb),tensor(ir),target


def collate(batch):
    rgb,ir,targets=zip(*batch)
    return torch.stack(rgb),torch.stack(ir),list(targets)


def move_targets(targets,device):
    return [{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]


def invert_boxes(boxes,transform):
    b=np.asarray(boxes,dtype=float).reshape(-1,5).copy()
    if len(b):
        b[:,:2]-=np.asarray(transform['offset']);b[:,:4]/=transform['scale']
    return b
