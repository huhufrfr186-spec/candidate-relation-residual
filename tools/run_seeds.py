import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crrl.config import load_config
from crrl.engine import train,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43,61,83]);args=p.parse_args()
    for seed in args.seeds:
        cfg=load_config(args.config);cfg['seed']=seed;run=Path(args.output)/f'seed_{seed}'
        last=train(cfg,run)
        checkpoint=run/'best.pt' if (run/'best.pt').exists() else last
        evaluate(checkpoint,cfg['data']['test'],run/'test',cfg.get('device','auto'))


if __name__=='__main__':main()
