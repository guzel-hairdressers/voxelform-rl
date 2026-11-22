import copy, functools, math, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.distributions import Normal
from voxelform.environment import Environment
from voxelform.patch_policy import PatchConfig, PatchFacades, PatchLearner
from voxelform.workload import action_reward

torch.set_num_threads(1)


def make(allocation="quota", device="cpu", **kwargs):
    c = PatchConfig(allocation=allocation, batched=True, **kwargs)
    e = Environment(
        (4, 12, 16),
        sun_count=4,
        device=device,
        frontier_factory=functools.partial(
            PatchFacades, config=c, depth_device=device if device != "cpu" else None
        ),
    )
    return e, PatchLearner(device, config=c)


class WorkloadTests(unittest.TestCase):
    def test_initial_budget_uses_actual_capacity_and_optional_cap(self):
        for cap in (0, 16):
            e, p = make(maximum_edit_voxels=cap)
            n = int(e.frontier.remove_skin.sum())
            m = e.count - e.target_count
            self.assertEqual(e.workload.initial_capacity, n)
            self.assertEqual(
                e.workload.limit, math.ceil(m / min(n * 0.5, cap or math.inf)) + 2
            )
            self.assertEqual(
                e.workload.quota(m, 0), math.ceil(m / (e.workload.limit - 2))
            )

    def test_deadline_does_not_extend_as_frontier_changes(self):
        e, p = make()
        original = copy.deepcopy(e.workload)
        a = p.choose(e, False)
        self.assertIsNotNone(a)
        e.apply(a, audit=False)
        self.assertEqual(e.workload.limit, original.limit)
        self.assertEqual(e.workload.initial_capacity, original.initial_capacity)
        self.assertEqual(e.workload.left(e.edits), original.limit - 1)

    def test_deep_permissions_do_not_inflate_deadline_capacity(self):
        c = PatchConfig(
            allocation="feedback", eligibility="depth", budget_reserve_fraction=0.2
        )
        e = Environment(
            (4, 40, 48),
            sun_count=4,
            frontier_factory=functools.partial(PatchFacades, config=c),
        )
        n = int(e.frontier.remove_skin.sum())
        base = math.ceil((e.count - e.target_count) / (n * 0.5))
        self.assertLess(n, int(e.frontier.eligible(-1).sum()))
        self.assertEqual(e.workload.initial_capacity, n)
        self.assertEqual(e.workload.limit, base + 2 + math.ceil(base * 0.2))

    def test_feedback_compensates_for_pruning_without_extending_deadline(self):
        e, p = make("feedback")
        p._operation = -1
        p.fields(e)
        scores = torch.ones(e.shape)
        legal = torch.as_tensor(e.frontier.remove_skin)
        a = p.allocation_counts(scores, legal, e.count - e.target_count)[0]
        limit = e.workload.limit
        e.workload.observe(10, 100)
        b = p.allocation_counts(scores, legal, e.count - e.target_count)[0]
        self.assertGreater(int(b[0]), int(a[0]))
        self.assertEqual(limit, e.workload.limit)

    def test_small_tail_exception_keeps_maximum_patch_count(self):
        e, p = make("feedback", minimum_patch=4, maximum_patches=4)
        mask = np.zeros(e.shape, bool)
        mask[0, 0, ::3] = True
        self.assertEqual(int(e.frontier._patch_filter(mask, 16).sum()), 4)
        self.assertEqual(int(e.frontier._patch_filter(mask, 17).sum()), 0)

    def test_retry_reduces_requested_amount_but_preserves_final_cap(self):
        e, p = make("quota")
        p._operation = -1
        p.fields(e)
        scores = torch.ones(e.shape)
        legal = torch.as_tensor(e.frontier.remove_skin)
        first = p.allocation_counts(scores, legal, 100)[0]
        last = p.allocation_counts(scores, legal, 100, retry_offset=7)[0]
        self.assertLess(int(last[0]), int(first[0]))
        self.assertLessEqual(int(first[0]), 100)

    def test_timeout_is_failure_and_has_negative_reward(self):
        e, p = make()
        e.workload.limit = 1
        # Deliberately retain only one safe selected voxel to exercise timeout.
        a = p.choose(e, False)
        self.assertIsNotNone(a)
        raw = np.zeros(e.shape, bool)
        raw[-1, 0, 0] = True
        a = e.frontier.project_mask(raw, None, -1, 1)
        before = e.far_error
        e.apply(a, audit=False)
        self.assertEqual(e.termination, "step_budget_exhausted")
        self.assertLess(action_reward(e, before), -1)
        with self.assertRaises(ValueError):
            e.apply(a)

    def test_success_on_last_step_beats_timeout_and_no_overshoot(self):
        e, p = make()
        e.configure_far((e.count - 1) / e.plot_area)
        e.workload.limit = 1
        a = p.choose(e, False)
        self.assertEqual(a.count, 1)
        before = e.far_error
        e.apply(a, audit=False)
        self.assertEqual(e.termination, "far_reached")
        self.assertGreater(action_reward(e, before), 1)

    def test_addition_uses_its_own_current_eligibility(self):
        e, p = make()
        occupied = e.allowed.clone()
        occupied[2:] = False
        e.restore_occupancy(occupied)
        self.assertEqual(e.workload.initial_capacity, int(e.frontier.add_skin.sum()))
        a = p.choose(e, False)
        self.assertEqual(a.operation, 1)
        self.assertFalse((a.mask & ~e.frontier.add_skin).any())

    def test_amount_head_receives_score_gradient_and_checkpoint_resumes(self):
        e, p = make("learned")
        torch.manual_seed(44)
        a = p.choose(e, True)
        self.assertIsNotNone(a)
        before = e.far_error
        e.apply(a, audit=False)
        p.record(action_reward(e, before))
        old = p.model.amount_head.weight.detach().clone()
        p.finish(p.bootstrap(e))
        self.assertGreater(
            float((p.model.amount_head.weight.detach() - old).abs().sum()), 0
        )
        q = PatchLearner()
        q.load(copy.deepcopy(p.state()))
        self.assertEqual(q.config, p.config)
        for k, v in p.model.state_dict().items():
            torch.testing.assert_close(q.model.state_dict()[k], v)
        self.assertIsNotNone(q.choose(e, True))

    def test_consumed_retry_scores_include_amount_latent(self):
        e, p = make("learned")
        original = e.frontier.project_mask

        def fixed(d, sample_shape=torch.Size()):
            return (d.loc.detach() + 0.2).expand(*sample_shape, *d.loc.shape).clone()

        with patch.object(Normal, "sample", fixed):
            p.choose(e, True)
        one = p.tape[-1][0].detach().clone()
        p.clear()
        calls = [0]

        def delayed(*args):
            calls[0] += 1
            return None if calls[0] < 3 else original(*args)

        with (
            patch.object(Normal, "sample", fixed),
            patch.object(e.frontier, "project_mask", delayed),
        ):
            p.choose(e, True)
        self.assertEqual(p.generation_attempts, 3)
        torch.testing.assert_close(p.tape[-1][0], 3 * one)

    def test_autosave_restores_fixed_deadline_not_recomputed_capacity(self):
        from voxelform.session import Session

        class Library:
            def catalog(self):
                return [{"site_id": "unused"}]

        with tempfile.TemporaryDirectory() as directory:
            e, p = make("learned")
            s = Session(env=e, autostart=False)
            s.policy = p
            s.persistence_root = Path(directory)
            s.start(training=True, single=True)
            s.advance_one()
            s.learn_rollout()
            s.pause()
            q = Session(library=Library(), autostart=False, device="cpu")
            q.persistence_root = Path(directory)
            q.build()
            self.assertEqual(q.env.workload, e.workload)
            self.assertEqual(q.env.edits, e.edits)
            torch.testing.assert_close(q.env.occupied, e.occupied)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_native_gpu_amount_sampling_backward(self):
        e, p = make("learned", "mps")
        a = p.choose(e, True)
        self.assertIsNotNone(a)
        before = e.far_error
        e.apply(a, audit=False)
        p.record(action_reward(e, before))
        stats = p.finish(p.bootstrap(e))
        self.assertGreater(stats["gradientNorm"], 0)
        self.assertEqual(e.frontier.depth_tensor.device.type, "mps")

    def test_invalid_options_rejected(self):
        for kw in (
            {"budget_fraction": 0},
            {"budget_fraction": float("nan")},
            {"budget_margin": -1},
            {"budget_margin": True},
            {"allocation": "x"},
            {"allocation": "learned", "mode": "smooth"},
        ):
            with self.assertRaises(ValueError):
                PatchConfig(**kw)


if __name__ == "__main__":
    unittest.main()
