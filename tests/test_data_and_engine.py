import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from crrl.prepare import prepare_pairs,prepare_vedai,minimum_rectangle,split_manifest
from crrl.data import PairedDataset,invert_boxes
from crrl.config import load_config
from crrl.engine import train,load_checkpoint


class DataEngineTests(unittest.TestCase):
    def test_native_voc_preparation_and_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['rgb','ir','labels']:(root/name).mkdir()
            for modality in ['rgb','ir']:Image.new('RGB',(100,50)).save(root/modality/'one.png')
            (root/'labels/one.xml').write_text('<annotation><object><name>car</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>20</ymax></bndbox></object></annotation>')
            prepare_pairs(root/'rgb',root/'ir',root/'labels',root/'labels',['car'],root/'manifest.json')
            dataset=PairedDataset(root/'manifest.json',(64,64));rgb,ir,target=dataset[0]
            self.assertEqual(tuple(rgb.shape),(3,64,64))
            np.testing.assert_allclose(invert_boxes(target['boxes'].numpy(),target['transform']),[[20,15,20,10,0]],atol=1e-5)

    def test_vedai_native_polygon(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'images';images.mkdir()
            for suffix in ['co','ir']:Image.new('RGB',(64,64)).save(images/f'00000001_{suffix}.png')
            # id, center x/y, heading, four x vertices, four y vertices, class, flags.
            annotation=root/'annotation512.txt'
            annotation.write_text('1 20 20 0 10 30 30 10 15 15 25 25 1 0 0\n')
            prepare_vedai(images,annotation,root/'manifests')
            data=json.loads((root/'manifests/all.json').read_text())
            np.testing.assert_allclose(data['samples'][0]['annotations'][0]['bbox'][:4],[20,20,20,10],atol=1e-8)

    def test_scene_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);samples=[]
            for scene in range(6):
                for j in range(2):samples.append({'id':f'{scene}-{j}','scene':str(scene),'rgb':'r.png','ir':'i.png','annotations':[],'ir_annotations':[]})
            path=root/'all.json';path.write_text(json.dumps({'classes':['object'],'samples':samples}))
            split_manifest(path,root/'split',17,.3)
            a=json.loads((root/'split/train.json').read_text())['samples'];b=json.loads((root/'split/val.json').read_text())['samples']
            self.assertFalse({s['scene'] for s in a}&{s['scene'] for s in b})
            self.assertEqual(len(a)+len(b),12)

    def test_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'image.png';Image.new('RGB',(32,32),(80,80,80)).save(image)
            annotation=[{'category_id':0,'bbox':[16,16,8,6,0]}]
            manifest=root/'train.json';manifest.write_text(json.dumps({'classes':['object'],'samples':[{'id':'one','rgb':str(image),'ir':str(image),'annotations':annotation,'ir_annotations':annotation}]}))
            cfg=load_config(Path(__file__).resolve().parents[1]/'configs/smoke.yaml');cfg['data']['train']=str(manifest);cfg['cpu_threads']=1
            checkpoint=train(cfg,root/'run');model,saved,state=load_checkpoint(checkpoint,torch.device('cpu'))
            self.assertEqual(state['step'],1)
            cfg['train']['epochs']=2;train(cfg,root/'run',checkpoint)
            _,_,state=load_checkpoint(checkpoint,torch.device('cpu'));self.assertEqual(state['step'],2);self.assertEqual(state['epoch'],1)


if __name__=='__main__':unittest.main()
