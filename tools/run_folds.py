import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crrl.config import load_config
from crrl.engine import train,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/vedai.yaml');p.add_argument('--manifests',default='data/vedai');p.add_argument('--output',required=True)
    args=p.parse_args()
    for fold in range(10):
        cfg=load_config(args.config)
        for split in ['train','val','test']:cfg['data'][split]=str(Path(args.manifests)/f'fold_{fold}_{split}.json')
        run=Path(args.output)/f'fold_{fold}';last=train(cfg,run)
        evaluate(run/'best.pt' if (run/'best.pt').exists() else last,cfg['data']['test'],run/'test',cfg.get('device','auto'))


if __name__=='__main__':main()
