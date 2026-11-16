import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull


def minimum_rectangle(points):
    p=np.asarray(points,dtype=float).reshape(-1,2)
    if len(p)<3:raise ValueError('At least three polygon vertices are required')
    hull=p[ConvexHull(p).vertices]
    edges=np.roll(hull,-1,axis=0)-hull
    angles=np.unique(np.arctan2(edges[:,1],edges[:,0]) % (math.pi/2))
    best=None
    for theta in angles:
        c,s=math.cos(theta),math.sin(theta);rotation=np.array([[c,-s],[s,c]])
        local=p@rotation;lo,hi=local.min(0),local.max(0);size=hi-lo
        candidate=(float(size.prod()),theta,(lo+hi)/2@rotation.T,size)
        if best is None or candidate[0]<best[0]:best=candidate
    _,theta,center,size=best
    if min(size)<=0:raise ValueError('Bounding boxes must have positive dimensions')
    return [float(center[0]),float(center[1]),float(size[0]),float(size[1]),float(theta)]


def parse_annotation(path,classes):
    path=Path(path);lookup={name.lower().replace(' ','_').replace('-','_'):i for i,name in enumerate(classes)}
    result=[]
    if path.suffix.lower()=='.xml':
        root=ET.parse(path).getroot()
        for obj in root.findall('object'):
            name=obj.findtext('name','').strip().lower().replace(' ','_').replace('-','_')
            if name not in lookup:continue
            rb=obj.find('robndbox');bb=obj.find('bndbox')
            if rb is not None:
                box=[float(rb.findtext(k)) for k in ['cx','cy','w','h','angle']]
            elif bb is not None and bb.find('x1') is not None:
                box=minimum_rectangle([[float(bb.findtext(f'x{i}')),float(bb.findtext(f'y{i}'))] for i in range(1,5)])
            elif bb is not None:
                x1,y1,x2,y2=[float(bb.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                box=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,0.]
            else:
                polygon=obj.find('polygon')
                if polygon is None:raise ValueError(f'Unknown annotation in {path}')
                box=minimum_rectangle([[float(polygon.findtext(f'x{i}')),float(polygon.findtext(f'y{i}'))] for i in range(1,5)])
            if min(box[2:4])<=0:raise ValueError(f'Invalid box in {path}')
            if obj.findtext('difficult','0')=='1':continue
            result.append({'category_id':lookup[name],'bbox':box})
    else:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            fields=line.split()
            if not fields or fields[0].lower().startswith(('imagesource','gsd')):continue
            if len(fields)<9:raise ValueError(f'Expected a polygon label in {path}: {line}')
            name=fields[8].lower().replace(' ','_').replace('-','_')
            if name not in lookup:continue
            if len(fields)>9 and int(fields[9])==1:continue
            result.append({'category_id':lookup[name],'bbox':minimum_rectangle(np.asarray(fields[:8],dtype=float).reshape(4,2))})
    return result


def prepare_pairs(rgb_dir,ir_dir,rgb_labels,ir_labels,classes,output,ids=None,crop_border=0):
    rgb_dir,ir_dir,rgb_labels,ir_labels=map(Path,[rgb_dir,ir_dir,rgb_labels,ir_labels])
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    def images(directory):return {p.stem:p for p in directory.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']}
    rgb,ir=images(rgb_dir),images(ir_dir)
    selected=sorted(rgb) if ids is None else ids
    samples=[]
    for image_id in selected:
        if image_id not in rgb or image_id not in ir:raise FileNotFoundError(f'Missing pair: {image_id}')
        annotations=[]
        for directory in [rgb_labels,ir_labels]:
            candidates=[directory/(image_id+suffix) for suffix in ['.xml','.txt']]
            label=next((p for p in candidates if p.exists()),None)
            if label is None:raise FileNotFoundError(f'Missing label for {image_id} in {directory}')
            annotations.append(parse_annotation(label,classes))
        paths=[]
        for modality,image_path,labels in zip(['rgb','ir'],[rgb[image_id],ir[image_id]],annotations):
            if crop_border:
                image=Image.open(image_path);w,h=image.size
                if min(w,h)<=2*crop_border:raise ValueError('Crop border exceeds image dimensions')
                dest=output.parent/'images'/modality/(image_id+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
                image.crop((crop_border,crop_border,w-crop_border,h-crop_border)).save(dest)
                for a in labels:a['bbox'][0]-=crop_border;a['bbox'][1]-=crop_border
                paths.append(str(dest))
            else:paths.append(str(image_path.resolve()))
        samples.append({'id':image_id,'rgb':paths[0],'ir':paths[1],'annotations':annotations[0],'ir_annotations':annotations[1]})
    output.write_text(json.dumps({'classes':classes,'samples':samples},indent=2),encoding='utf-8')
    return len(samples)


def split_manifest(path,output,seed=17,validation_fraction=.1,folds=None):
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    for sample in data['samples']:
        for field in ['rgb','ir']:
            p=Path(sample[field]);sample[field]=str(p if p.is_absolute() else path.parent/p)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups={}
    for sample in data['samples']:groups.setdefault(sample.get('scene',sample['id']),[]).append(sample)
    keys=sorted(groups);np.random.default_rng(seed).shuffle(keys)
    def write(name,indices):
        result={'classes':data['classes'],'samples':[s for k in indices for s in groups[k]]}
        (output/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if folds:
        for fold,test_keys in enumerate(np.array_split(np.asarray(keys,dtype=object),folds)):
            test_keys=list(test_keys);remaining=[k for k in keys if k not in test_keys]
            n=max(1,round(len(remaining)*validation_fraction))
            write(f'fold_{fold}_test',test_keys);write(f'fold_{fold}_val',remaining[:n]);write(f'fold_{fold}_train',remaining[n:])
    else:
        n=max(1,round(len(keys)*validation_fraction))
        write('val',keys[:n]);write('train',keys[n:])


def prepare_vedai(images,annotations,output,fold_mat=None):
    from scipy.io import loadmat
    images=Path(images).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    category_ids=[1,2,4,5,9,10,11,23,31]
    classes=['car','truck','tractor','camping_car','van','other','pickup','boat','plane']
    records=np.loadtxt(annotations,ndmin=2)
    by_id={}
    for row in records:
        category=int(row[12])
        if category not in category_ids:continue
        image_id=f'{int(row[0]):08d}'
        points=np.stack((row[4:8],row[8:12]),-1)
        by_id.setdefault(image_id,[]).append({'category_id':category_ids.index(category),'bbox':minimum_rectangle(points)})
    samples=[]
    for rgb in sorted(images.glob('*_co.png')):
        image_id=rgb.stem[:-3];ir=images/(image_id+'_ir.png')
        if not ir.is_file():raise FileNotFoundError(ir)
        objects=by_id.get(image_id,[])
        samples.append({'id':image_id,'rgb':str(rgb),'ir':str(ir),'annotations':objects,'ir_annotations':objects})
    if not samples:raise ValueError(f'No VEDAI image pairs found in {images}')
    def write(name,selection):
        (output/(name+'.json')).write_text(json.dumps({'classes':classes,'samples':selection},indent=2),encoding='utf-8')
    write('all',samples)
    if fold_mat:
        fold=np.asarray(loadmat(fold_mat)['fold'])
        if fold.shape[0]!=10 and fold.shape[1]==10:fold=fold.T
        if fold.shape[0]!=10:raise ValueError('VEDAI fold.mat must contain ten fold rows')
        sets=[{f'{int(v):08d}' for v in row if np.isfinite(v)} for row in fold]
        available={s['id'] for s in samples}
        for ids in sets:
            missing=ids-available
            if missing:raise ValueError(f'Fold image IDs not found: {sorted(missing)[:5]}')
        for i,test in enumerate(sets):
            validation=sets[(i+1)%10]
            write(f'fold_{i}_test',[s for s in samples if s['id'] in test])
            write(f'fold_{i}_val',[s for s in samples if s['id'] in validation])
            write(f'fold_{i}_train',[s for s in samples if s['id'] not in test|validation])
    else:split_manifest(output/'all.json',output,folds=10)
    return len(samples)
