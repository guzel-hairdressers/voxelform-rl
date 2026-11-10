import unittest
import numpy as np
import torch
from tools.architecture_hypotheses import ExperimentalNetwork,ExperimentalLearner,VARIANTS,inputs,prepare,train
from voxelform.policy import Learner
from voxelform.environment import Environment

torch.set_num_threads(1)
class HypothesisMechanisms(unittest.TestCase):
    def test_experimental_baseline_exactly_matches_live_forward(self):
        env=Environment((4,8,10),sun_count=4)
        old=Learner();new=ExperimentalLearner(dict(single=False,facade=False,encoder='2d'))
        for a,b in zip(old.fields(env),new.fields(env)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_single_head_computes_one_channel_and_operation_is_trainable(self):
        model=ExperimentalNetwork(dict(single=True,facade=False,encoder='2d'));seen=[]
        hook=model.mask_head[-1].register_forward_hook(lambda m,i,o:seen.append(o.shape[1]))
        x=torch.randn(4,5,8,10);g=torch.tensor([1.,.75,.25]);out,_=model(x,g,-1)
        out[0].sum().backward();self.assertEqual(seen,[1]);self.assertGreater(float(model.goal_encoder.weight.grad[:,-1].abs().sum()),0)
        hook.remove()
    def test_facade_channels_are_exact_current_snapshot_and_learnable(self):
        env=Environment((4,8,10),sun_count=4);x=inputs(env,True)
        np.testing.assert_array_equal(x[:,5].numpy().astype(bool),env.frontier.remove_skin)
        np.testing.assert_array_equal(x[:,6].numpy().astype(bool),env.frontier.add_skin)
        p=ExperimentalLearner(dict(single=False,facade=True,encoder='2d'));p.fields(env)[0].sum().backward()
        self.assertGreater(float(p.model.encoder[0].weight.grad[:,5].abs().sum()),0)
    def test_3d_and_multiscale_enable_distinct_information_paths(self):
        def gradient(kind):
            m=ExperimentalNetwork(dict(single=False,facade=False,encoder=kind));torch.manual_seed(99)
            x=torch.rand(6,5,32,32,requires_grad=True);y,_=m(x,torch.tensor([1.,.75,.25]),-1)
            y[0,2,16,16].backward();return x.grad
        a,b,c=[gradient(k) for k in ('2d','3d','multiscale')]
        self.assertEqual(float(a[3].abs().sum()),0);self.assertGreater(float(b[3].abs().sum()),0)
        self.assertEqual(float(b[:,:,16,8].abs().sum()),0);self.assertGreater(float(c[:,:,16,8].abs().sum()),0)
    def test_training_exposure_is_equal_for_each_plot_and_operation(self):
        envs=[Environment((4,8,10),sun_count=4,seed=seed) for seed in (11,22)]
        p=ExperimentalLearner(dict(single=True,facade=True,encoder='3d'))
        result=train(p,envs,123,32)
        self.assertEqual(result['attemptedActions'],32)
        self.assertEqual([x['attemptedActions'] for x in result['conditionCounts']],[8,8,8,8])
        self.assertTrue(result['updates'])
    def test_all_combinations_keep_actual_facade_and_far_constraints(self):
        env=Environment((4,8,10),sun_count=4)
        for config in VARIANTS:
            for operation in ('subtract','add'):
                with self.subTest(config=config,operation=operation):
                    prepare(env,operation);p=ExperimentalLearner(config)
                    a=p.choose(env,training=True);self.assertIsNotNone(a)
                    self.assertFalse((a.mask&~env.frontier.eligible(a.operation)).any())
                    self.assertTrue(env.frontier.validate_mask(a.mask,a.operation));self.assertGreaterEqual(a.count,2)
                    before=env.far_error;env.apply(a,audit=False);p.record(before-env.far_error);stats=p.finish(p.bootstrap(env))
                    self.assertGreater(stats['gradientNorm'],0);self.assertEqual(stats['directMaskUpdates'],1)
if __name__=='__main__':unittest.main()
