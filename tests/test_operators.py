import itertools
import math
import unittest
import numpy as np
import torch
from crrl.graph import reduced_incidence,weighted_projection,soft_transport,BoundedCorrection,local_graphs
from crrl.probability import log_normal_interval,angular_log_mass,mixture_region_log_prob
from crrl.assignment import lexicographic_assignment,assign_targets
from crrl.decode import region_candidates,score_regions
from crrl.geometry import rotated_iou
from crrl.metrics import evaluate_detection


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(17)

    def test_projection_invariance_and_weighted_orthogonality(self):
        edges=[[0,0],[0,1],[1,0],[1,1],[2,2]]
        b,r,k=reduced_incidence(edges,3,3)
        h=torch.randn(5,3,dtype=torch.double);p=torch.rand(5,dtype=torch.double)
        z=weighted_projection(h,p,r);w=.05+.95*p
        self.assertLess((b.T@(w[:,None]*z)).abs().max().item(),1e-12)
        addition=b@torch.randn(6,3,dtype=torch.double)
        torch.testing.assert_close(z,weighted_projection(h+addition,p,r),atol=1e-12,rtol=1e-12)
        self.assertEqual(len(edges)-b.shape[1]+k,1)
        self.assertLess(z[-1].abs().max().item(),1e-12)

    def test_projection_gradient(self):
        _,r,_=reduced_incidence([[0,0],[0,1],[1,0],[1,1]],2,2)
        h=torch.randn(4,2,dtype=torch.double,requires_grad=True)
        p=torch.rand(4,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda h,p:weighted_projection(h,p,r),(h,p)))

    def test_transport_marginals_and_gradient(self):
        support=torch.tensor([[1,0,1],[1,1,0]],dtype=torch.bool)
        s=(torch.randn(2,3,dtype=torch.double)*.1).requires_grad_();dust=torch.tensor(.2,dtype=torch.double,requires_grad=True)
        p=soft_transport(s,support,dust,.5,200)
        torch.testing.assert_close(p.sum(-1),torch.tensor([1.,1.,3.],dtype=torch.double),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(p.sum(-2),torch.tensor([1.,1.,1.,2.],dtype=torch.double),atol=1e-10,rtol=1e-10)
        self.assertEqual(p[0,1].item(),0.)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:soft_transport(a,support,b,.5,100),(s,dust)))

    def test_empty_transport(self):
        p=soft_transport(torch.empty(2,0),torch.empty(2,0,dtype=torch.bool),torch.tensor(0.))
        torch.testing.assert_close(p[:,0],torch.tensor([1.,1.,0.]))
        p=soft_transport(torch.zeros(2,3),torch.zeros(2,3,dtype=torch.bool),torch.tensor(0.))
        torch.testing.assert_close(p[:2,3],torch.ones(2))
        torch.testing.assert_close(p[2,:3],torch.ones(3))

    def test_composed_transport_projection_probability_gradient(self):
        _,design,_=reduced_incidence([[0,0],[0,1],[1,0],[1,1]],2,2)
        matrix=torch.randn(2,12,dtype=torch.double)*.05
        h=torch.randn(4,2,dtype=torch.double,requires_grad=True)
        s=(torch.randn(2,2,dtype=torch.double)*.1).requires_grad_()
        def loss(features,scores):
            plan=soft_transport(scores,torch.ones(2,2,dtype=torch.bool),scores.new_tensor(.1),.5,60)
            p=plan[:2,:2].flatten();z=weighted_projection(features,p,design)
            value=features@matrix+.8*torch.tanh(z@matrix)
            value=value.reshape(2,2,12)
            raw=torch.cat((value,value.new_zeros(2,1,12)),1)
            masses=torch.cat((plan[:2,:2],plan[:2,2:]),-1)
            return -mixture_region_log_prob(raw,masses,torch.zeros(2,dtype=torch.long),torch.zeros(2,5,dtype=torch.long),1,[.1,.1,.1,.1,math.pi/90],{}).mean()
        self.assertTrue(torch.autograd.gradcheck(loss,(h,s),eps=1e-6,atol=1e-5,rtol=1e-3))

    def test_correction_and_parameter_count(self):
        correction=BoundedCorrection(32,56,.8)
        self.assertEqual(sum(p.numel() for p in correction.parameters()),5752)
        self.assertEqual(sum(p.numel() for p in BoundedCorrection(96,56,.8,37).parameters()),5717)
        self.assertTrue(torch.equal(correction(torch.zeros(4,32)),torch.zeros(4,56)))
        self.assertLessEqual(correction(torch.randn(4,32)*100).abs().max().item(),.800001)

    def test_regular_graph_cycle_dimension(self):
        groups=local_graphs(8,8,2,2)
        interior=next(g for g in groups if g.support.shape==(4,36))
        self.assertEqual(len(interior.edges),100)
        self.assertEqual(interior.design.shape,(100,39))

    def test_gaussian_tails(self):
        a=torch.tensor([-41.,-1.,1.,40.],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.isfinite(log_normal_interval(a,a+.1)).all())
        self.assertTrue(torch.autograd.gradcheck(lambda x:log_normal_interval(x,x+.1),(a,)))

    def test_periodic_partition_and_gradient(self):
        lo=torch.arange(90,dtype=torch.double)*math.pi/90;hi=lo+math.pi/90
        for k in [0.,1.,100.]:
            mu=torch.tensor(.83,dtype=torch.double)
            masses=angular_log_mass(lo,hi,mu,torch.tensor(k,dtype=torch.double)).exp()
            self.assertAlmostEqual(float(masses.sum()),1.,places=11)
        mu=torch.tensor(.2,dtype=torch.double,requires_grad=True);k=torch.tensor(3.,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda m,c:angular_log_mass(m.new_tensor(0.),m.new_tensor(.1),m,c),(mu,k)))

    def test_lexicographic_assignment(self):
        for cost in [np.zeros((3,5)),np.array([[1.,1.,3.],[1.,1.,0.]]),np.array([[0.,0.],[0.,0.]])]:
            expected=min(itertools.permutations(range(cost.shape[1]),cost.shape[0]),key=lambda p:(sum(cost[i,j] for i,j in enumerate(p)),p))
            self.assertEqual(tuple(lexicographic_assignment(cost)),expected)

    def test_assignment_coverage(self):
        points=torch.tensor([[.1,.1],[.5,.5],[.9,.9]])
        boxes=torch.tensor([[50.,50.,1.,1.,0.],[50.,50.,1.,1.,0.]])
        owners=assign_targets(points,boxes,(100,100))
        self.assertTrue(all((owners==i).any() for i in range(2)))
        with self.assertRaises(ValueError):assign_targets(points,boxes.repeat(2,1),(100,100))

    def test_uniform_proposal_invariance(self):
        mu=torch.zeros(1,4,dtype=torch.double);k=torch.zeros(1,dtype=torch.double);widths=[.1,.1,.1,.1,math.pi/6]
        a=region_candidates(mu,torch.tensor([0.]),k,widths)
        b=region_candidates(mu,torch.tensor([1.1]),k,widths)
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),81*6)

    def test_component_splitting_invariance(self):
        mean=torch.zeros(2,4,dtype=torch.double);sigma=torch.ones_like(mean)*.1
        angle=torch.tensor([.2,.5],dtype=torch.double);k=torch.ones(2,dtype=torch.double)
        widths=[.1,.1,.1,.1,math.pi/18];regions=region_candidates(mean,angle,k,widths)[:50]
        score=score_regions((mean,sigma,angle,k),torch.tensor([.4,.6],dtype=torch.double).log(),regions,widths)
        ix=torch.tensor([0,0,1]);split=score_regions((mean[ix],sigma[ix],angle[ix],k[ix]),torch.tensor([.2,.2,.6],dtype=torch.double).log(),regions,widths)
        torch.testing.assert_close(score,split,atol=1e-12,rtol=1e-12)

    def test_rotated_geometry_and_ap(self):
        box=[10.,10.,8.,4.,.4]
        self.assertAlmostEqual(rotated_iou(box,box),1.,places=10)
        self.assertEqual(rotated_iou(box,[100.,100.,8.,4.,.4]),0.)
        gt=[{'id':'a','annotations':[{'bbox':box,'category_id':0}]}]
        predictions=[{'id':'a','boxes':[box,box],'scores':[.9,.8],'labels':[0,0]}]
        result=evaluate_detection(predictions,gt,['object'])
        self.assertAlmostEqual(result['AP50:95'],100.,places=8)


if __name__=='__main__':unittest.main()
