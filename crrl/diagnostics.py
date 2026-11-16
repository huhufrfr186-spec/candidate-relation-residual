import json
from pathlib import Path
import numpy as np
from .data import read_manifest
from .geometry import rotated_iou


def diagnose(manifest,predictions,output,threshold=.01):
    data=read_manifest(manifest);rows=[];tp=fp=fn=0
    pred={str(x['id']):x for x in json.loads(Path(predictions).read_text(encoding='utf-8'))} if predictions else {}
    for sample in data['samples']:
        annotations=sample.get('annotations',[]);boxes=np.asarray([a['bbox'] for a in annotations],dtype=float).reshape(-1,5)
        distances=np.max(np.abs(boxes[:,None,:2]-boxes[None,:,:2]),axis=-1) if len(boxes) else np.empty((0,0))
        from PIL import Image
        width,height=Image.open(sample['rgb']).size
        # The square-window proxy is defined in the 640 x 512 image coordinate system.
        normalized=boxes[:,:2]*np.array([640/width,512/height])
        distance=np.max(np.abs(normalized[:,None,:]-normalized[None,:,:]),-1) if len(boxes) else np.empty((0,0))
        count=(distance<=16).sum(-1)-1
        for i,(box,a) in enumerate(zip(boxes,annotations)):
            rows.append({'id':sample['id'],'object_index':i,'category_id':a['category_id'],
                         'neighbors':int(count[i]),'competition_group':int(min(count[i],2)),'area':float(box[2]*box[3])})
        if str(sample['id']) in pred:
            record=pred[str(sample['id'])];matched=set()
            order=sorted(range(len(record['scores'])),key=lambda i:(-record['scores'][i],i))
            for i in order:
                if record['scores'][i]<threshold:continue
                candidates=[(rotated_iou(record['boxes'][i],a['bbox']),j) for j,a in enumerate(annotations)
                            if j not in matched and a['category_id']==record['labels'][i]]
                score,j=max(candidates,default=(-1,-1))
                if score>=.5:matched.add(j);tp+=1
                else:fp+=1
            fn+=len(annotations)-len(matched)
    result={'objects':rows,'fixed_operating_point':{'threshold':threshold,'IoU':.5,'TP':tp,'FP':fp,'FN':fn,
             'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'F1':2*tp/max(2*tp+fp+fn,1)}}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding='utf-8')
