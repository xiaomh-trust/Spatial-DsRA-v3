"""CPU verification of the two Spatial-DsRA implementations (no model weights needed)."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openvla-SF"))


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DSRA = load("openvla-SF/prismatic/models/spatial_dsra.py")
LORA = load("openpi-SF/src/openpi/models_pytorch/spatial_lora.py")


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DSRA.SpatialDsRA(16, 12, 3, bottleneck_dim=8, rank=3,
                                     warmup_steps=2, ramp_steps=2, freeze_step=5, beta_r=1e-4, eta=1e-4)
        self.hidden = [torch.randn(2, 8, 16, requires_grad=True) for _ in range(3)]
        self.target = torch.randn(2, 8, 12, requires_grad=True)

    def test_formula_and_mask(self):
        m = self.model
        m.set_step(4)
        with torch.no_grad():
            m.rho.normal_()
        mask = torch.tensor([[True] * 4 + [False] * 4, [True] * 8])
        loss, reg, _, reg_r, orth = m(self.hidden, self.target, mask)
        reference = []
        for i, hidden in enumerate(self.hidden):
            b = F.gelu(m.down(m.norm(hidden)))
            r = torch.diag(m.a_max * m.rho[i].tanh())
            u = m.up(b @ (torch.eye(8) + m.M @ r @ m.M.T))
            d = 1 - F.cosine_similarity(u, self.target, dim=-1)
            reference.append(d[mask].mean())
        torch.testing.assert_close(loss, torch.stack(reference).mean())
        torch.testing.assert_close(reg, m.beta_r * reg_r + m.eta * orth)

    def test_progressive_gradients_and_optimizer(self):
        m = self.model
        opt = torch.optim.AdamW(DSRA.parameter_groups(m, self.hidden, 0.01, 0.1))
        for step, stage in [(0, 1), (3, 2), (4, 2), (5, 3)]:
            opt.zero_grad(set_to_none=True)
            m.set_step(step)
            before = {n: p.detach().clone() for n, p in m.named_parameters()}
            loss, reg, *_ = m(self.hidden, self.target)
            (loss + reg).backward()
            self.assertEqual(m.stage, stage)
            self.assertTrue(all(h.grad is not None and h.grad.abs().sum() > 0 for h in self.hidden))
            self.assertIsNone(self.target.grad)
            self.assertEqual(m.rho.grad is None, stage != 2)
            self.assertEqual(m.M.grad is None, stage != 2)
            opt.step()
            if stage == 3:
                self.assertEqual(reg.item(), 0)
                for n, p in m.named_parameters():
                    torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            if step == 4:
                self.assertGreater(m.M.grad.abs().sum().item(), 0)

    def test_invalid_and_empty_masks(self):
        with self.assertRaises(ValueError):
            self.model(self.hidden[:-1], self.target)
        with self.assertRaises(ValueError):
            self.model(self.hidden, self.target[..., :-1])
        loss, _, *_ = self.model(self.hidden, self.target, torch.zeros(2, 8, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)

    def test_teacher_camera_order(self):
        features = torch.zeros(2, 2, 9, 12)
        features[:, 1] = 5
        target = DSRA.teacher_target(features, (3, 3), (2, 2))
        self.assertEqual(target.shape, (2, 8, 12))
        torch.testing.assert_close(target[:, 4:] - target[:, :4], torch.full((2, 4, 12), 5.0))

    def test_lora_merge(self):
        model = nn.Sequential(LORA.SpatialLoRALinear(nn.Linear(16, 12), 4))
        with torch.no_grad():
            model[0].lora_B.normal_()
        x = torch.randn(2, 8, 16)
        plain = nn.Sequential(nn.Linear(16, 12))
        plain.load_state_dict(LORA.deployment_state_dict(model))
        torch.testing.assert_close(model(x), plain(x), rtol=1e-5, atol=1e-6)

    def test_production_channels(self):
        for dv, layers in [(4096, 32), (2048, 18)]:
            m = DSRA.SpatialDsRA(dv, 2048, layers, freeze_step=100000, beta_r=1e-4, eta=1e-4)
            m.set_step(0)
            hidden = [torch.randn(1, 2, dv, requires_grad=True) for _ in range(layers)]
            loss, reg, per_layer, *_ = m(hidden, torch.randn(1, 2, 2048))
            (loss + reg).backward()
            self.assertEqual(per_layer.shape, (layers,))
            self.assertTrue(all(x.grad is not None for x in hidden))

    def test_branches_identical(self):
        self.assertEqual((ROOT / "openvla-SF/prismatic/models/spatial_dsra.py").read_bytes(),
                         (ROOT / "openpi-SF/src/openpi/models_pytorch/spatial_dsra.py").read_bytes())

    def test_cache_augmented_input_identity(self):
        class Teacher(nn.Module):
            patch_size = 2
            calls = 0

            def forward(self, images):
                self.calls += 1
                features = images.mean((-1, -2, -3))[..., None, None].expand(-1, -1, 5, 12)
                return dict(features=[features], patch_start_idx=1, images=images)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.bin"
            checkpoint.write_bytes(b"test-teacher-weights")
            teacher = Teacher()
            cache = DSRA.FrozenTeacherCache(Path(directory) / "cache", checkpoint)
            images = torch.randn(1, 2, 3, 4, 4)
            first, hw = cache.extract(teacher, images, -1)
            second, _ = cache.extract(teacher, images, -1)
            self.assertEqual(teacher.calls, 1)
            self.assertEqual(hw, (2, 2))
            torch.testing.assert_close(first, second)
            cache.extract(teacher, images + 0.1, -1)
            self.assertEqual(teacher.calls, 2)

    def test_ddp_stage_transitions(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest("Gloo unavailable")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=0, world_size=1)
            try:
                m = self.model
                ddp = torch.nn.parallel.DistributedDataParallel(m, find_unused_parameters=True)
                for step in (0, 1, 3, 4, 5, 6):
                    m.zero_grad(set_to_none=True)
                    m.set_step(step)
                    loss, reg, *_ = ddp(self.hidden, self.target)
                    (loss + reg).backward()
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
