import numpy as np
from .geometry import rotated_iou


def evaluate_detection(predictions,ground_truth,classes):
    by_id={str(x['id']):x for x in ground_truth}
    thresholds=np.linspace(.5,.95,10);aps=np.full((len(classes),10),np.nan)
    per_class={}
    for c,name in enumerate(classes):
        total=sum(sum(int(a['category_id'])==c and not a.get('ignore',False) for a in sample.get('annotations',[])) for sample in ground_truth)
        if total==0:continue
        rows=[]
        for prediction in predictions:
            for index,(box,score,label) in enumerate(zip(prediction['boxes'],prediction['scores'],prediction['labels'])):
                if label==c:rows.append((float(score),str(prediction['id']),index,box))
        rows.sort(key=lambda x:(-x[0],x[1],x[2]))
        for ti,threshold in enumerate(thresholds):
            matched={};tp=[];fp=[]
            for score,image_id,_,box in rows:
                if image_id not in by_id:raise ValueError(f'Prediction ID {image_id} is not in the ground truth')
                annotations=[a for a in by_id[image_id].get('annotations',[]) if int(a['category_id'])==c]
                used=matched.setdefault(image_id,set())
                ious=[rotated_iou(box,a['bbox']) if j not in used or a.get('ignore',False) else -1 for j,a in enumerate(annotations)]
                candidates=[j for j,a in enumerate(annotations) if not a.get('ignore',False) and ious[j]>=threshold]
                if not candidates:candidates=[j for j,a in enumerate(annotations) if a.get('ignore',False) and ious[j]>=threshold]
                j=max(candidates,key=lambda j:(ious[j],-j)) if candidates else -1
                if j>=0 and ious[j]>=threshold:
                    if annotations[j].get('ignore',False):continue
                    used.add(j);tp.append(1);fp.append(0)
                else:tp.append(0);fp.append(1)
            tp=np.cumsum(tp);fp=np.cumsum(fp);recall=tp/total;precision=tp/np.maximum(tp+fp,1)
            interpolated=[precision[recall>=r].max() if np.any(recall>=r) else 0. for r in np.linspace(0,1,101)]
            aps[c,ti]=np.mean(interpolated)*100
        per_class[name]={'AP50':float(aps[c,0]),'AP75':float(aps[c,5]),'AP50:95':float(aps[c].mean()),'objects':total}
    def mean(values):return float(np.nanmean(values)) if np.isfinite(values).any() else 0.
    return {'AP50':mean(aps[:,0]),'AP75':mean(aps[:,5]),'AP50:95':mean(aps),'per_class':per_class}
