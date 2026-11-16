import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .config import load_config


def main(argv=None):
    parser=argparse.ArgumentParser(prog='crrl',description='Candidate relation residual learning')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--seed',type=int);p.add_argument('--device')
    for name in ['evaluate','predict']:
        p=sub.add_parser(name);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='auto');p.add_argument('--intervention',choices=['zero','shuffle']);p.add_argument('--seed',type=int,default=17)
    p=sub.add_parser('benchmark');p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='auto');p.add_argument('--warmup',type=int,default=100);p.add_argument('--iterations',type=int,default=500)
    p=sub.add_parser('prepare');p.add_argument('--rgb-dir',required=True);p.add_argument('--ir-dir',required=True);p.add_argument('--rgb-labels',required=True);p.add_argument('--ir-labels',required=True);p.add_argument('--classes',required=True,help='Comma-separated category names');p.add_argument('--ids');p.add_argument('--crop-border',type=int,default=0);p.add_argument('--output',required=True)
    p=sub.add_parser('split');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--validation-fraction',type=float,default=.1);p.add_argument('--folds',type=int)
    p=sub.add_parser('prepare-vedai');p.add_argument('--images',required=True);p.add_argument('--annotations',required=True);p.add_argument('--fold-mat');p.add_argument('--output',required=True)
    p=sub.add_parser('summarize');p.add_argument('--metrics',nargs='+',required=True);p.add_argument('--paired-metrics',nargs='+');p.add_argument('--output',required=True)
    p=sub.add_parser('export');p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--classes',required=True)
    p=sub.add_parser('diagnostics');p.add_argument('--manifest',required=True);p.add_argument('--predictions');p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=.01)
    args=parser.parse_args(argv)
    if args.command=='train':
        from .engine import train
        cfg=load_config(args.config)
        if args.seed is not None:cfg['seed']=args.seed
        if args.device:cfg['device']=args.device
        train(cfg,args.output,args.resume)
    elif args.command in ['evaluate','predict']:
        from .engine import evaluate,load_checkpoint,loader_for,predict_loader,resolve_device,seed_all
        seed_all(args.seed)
        if args.command=='evaluate':evaluate(args.checkpoint,args.manifest,args.output,args.device,args.intervention)
        else:
            device=resolve_device(args.device);model,cfg,_=load_checkpoint(args.checkpoint,device)
            cfg['data']['test']=args.manifest;rows=predict_loader(model,loader_for(cfg,'test'),cfg,device,args.intervention)
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rows),encoding='utf-8')
    elif args.command=='benchmark':
        from .engine import benchmark
        if args.warmup<0 or args.iterations<1:parser.error('warmup must be nonnegative and iterations positive')
        benchmark(args.checkpoint,args.manifest,args.output,args.warmup,args.iterations,args.device)
    elif args.command=='prepare':
        from .prepare import prepare_pairs
        ids=Path(args.ids).read_text().split() if args.ids else None
        count=prepare_pairs(args.rgb_dir,args.ir_dir,args.rgb_labels,args.ir_labels,args.classes.split(','),args.output,ids,args.crop_border)
        print(json.dumps({'samples':count,'manifest':args.output}))
    elif args.command=='split':
        from .prepare import split_manifest
        if not 0<args.validation_fraction<1:parser.error('validation_fraction must be between zero and one')
        split_manifest(args.manifest,args.output,args.seed,args.validation_fraction,args.folds)
    elif args.command=='prepare-vedai':
        from .prepare import prepare_vedai
        print(json.dumps({'samples':prepare_vedai(args.images,args.annotations,args.output,args.fold_mat)}))
    elif args.command=='summarize':
        values=[json.loads(Path(p).read_text(encoding='utf-8'))['AP50:95'] for p in args.metrics]
        def stats(x):
            x=np.asarray(x,dtype=float);n=len(x);sd=float(x.std(ddof=1)) if n>1 else 0.
            half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else 0.
            return {'n':n,'mean':float(x.mean()),'sample_sd':sd,'95%_t_interval':[float(x.mean()-half),float(x.mean()+half)]}
        result={'AP50:95':stats(values)}
        if args.paired_metrics:
            paired=[json.loads(Path(p).read_text(encoding='utf-8'))['AP50:95'] for p in args.paired_metrics]
            if len(paired)!=len(values):parser.error('Paired metric lists must have equal length and matching seed order')
            result['paired_difference']=stats(np.asarray(values)-paired)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    elif args.command=='export':
        from .geometry import box_corners
        rows=json.loads(Path(args.predictions).read_text(encoding='utf-8'));classes=args.classes.split(',');output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        handles=[(output/(name+'.txt')).open('w',encoding='utf-8') for name in classes]
        try:
            for row in rows:
                for box,score,label in zip(row['boxes'],row['scores'],row['labels']):
                    coordinates=' '.join(f'{v:.6f}' for v in box_corners(box).flatten())
                    handles[label].write(f"{row['id']} {score:.10g} {coordinates}\n")
        finally:
            for h in handles:h.close()
    elif args.command=='diagnostics':
        from .diagnostics import diagnose
        diagnose(args.manifest,args.predictions,args.output,args.threshold)


if __name__=='__main__':main()
