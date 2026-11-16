import unittest
import torch
from crrl.model import CandidateRelationDetector
from crrl.losses import DetectionLoss
from crrl.decode import Decoder
from crrl.probability import mixture_region_log_prob


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.manual_seed(29)

    def test_train_step_both_heads_and_contexts(self):
        rgb=torch.rand(1,3,32,32);ir=torch.rand_like(rgb)
        target={'boxes':torch.tensor([[16.,16.,4.,6.,.2]]),'labels':torch.tensor([0]),
                'ir_boxes':torch.tensor([[17.,16.,4.,6.,.2]]),'ir_labels':torch.tensor([0])}
        for mode in ['continuous','probabilistic']:
            for context in ['none','graph','residual']:
                cfg={'classes':1,'channels':32,'output':mode,'context':context,'pretrained':False,
                     'radius':1,'block':2,'query_grid':[4,4],'sinkhorn_iterations':50,'temperature':.5}
                model=CandidateRelationDetector(cfg);pred=model(rgb,ir)
                self.assertEqual(pred['raw'].shape[:3],(1,16,10))
                torch.testing.assert_close(pred['mass'].sum(-1),torch.ones(1,16),atol=1e-5,rtol=1e-5)
                criterion=DetectionLoss(cfg,{'assignment':'unique'})
                result=criterion(pred,[target],(32,32));self.assertTrue(torch.isfinite(result['loss']))
                result['loss'].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if mode=='continuous':
                    result=Decoder(cfg,{}, {'score_threshold':0.,'max_detections':5})(pred,(32,32))
                    self.assertGreater(len(result[0]['boxes']),0)

    def test_empty_targets(self):
        cfg={'classes':1,'output':'continuous','pretrained':False,'radius':0,'query_grid':[4,4]}
        model=CandidateRelationDetector(cfg)
        pred=model(torch.rand(1,3,32,32),torch.rand(1,3,32,32))
        empty={'boxes':torch.empty(0,5),'labels':torch.empty(0,dtype=torch.long),'ir_boxes':torch.empty(0,5),'ir_labels':torch.empty(0,dtype=torch.long)}
        value=DetectionLoss(cfg,{})(pred,[empty],(32,32))['loss']
        self.assertTrue(torch.isfinite(value));value.backward()

    def test_probabilistic_decoder_and_refinement(self):
        cfg={'classes':1,'output':'probabilistic','pretrained':False,'radius':0,'query_grid':[1,1],'temperature':.5}
        model=CandidateRelationDetector(cfg).eval()
        with torch.no_grad():pred=model(torch.rand(1,3,32,32),torch.rand(1,3,32,32))
        for refine in [False,True]:
            decoder=Decoder(cfg,{}, {'score_threshold':1e-10,'max_detections':5,'refine':refine,'refinement_steps':3})
            result=decoder(pred,(32,32))[0]
            self.assertGreater(len(result['boxes']),0)
            self.assertTrue(torch.isfinite(torch.tensor(result['boxes'])).all())


if __name__=='__main__':unittest.main()
