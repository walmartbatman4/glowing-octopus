"""
Unit tests for invertible transforms.
"""

import pytest
import torch
from unfold.transforms import (
    DequantizeTransform,
    IdentityTransform,
    Logit01Transform,
    LogTransform,
    Pipeline,
    StandardizeTransform,
    ZeroInflatedLogitTransform,
)


def test_identity_transform():
    t = IdentityTransform()
    x = torch.randn(50, dtype=torch.float32)
    y, ldj = t(x)
    x_rec = t.inverse(y)
    assert torch.allclose(x, x_rec, atol=1e-6)
    assert torch.allclose(ldj, torch.zeros_like(x))


def test_log_transform_roundtrip_and_jacobian():
    t = LogTransform(eps=1e-3)
    x = torch.linspace(0.01, 100.0, 50, requires_grad=True, dtype=torch.float32)
    y, ldj = t(x, training=False)
    x_rec = t.inverse(y)
    assert torch.allclose(x, x_rec, atol=1e-5)

    # Check analytical Jacobian against autograd
    grad_y = torch.autograd.grad(y.sum(), x)[0]
    expected_ldj = torch.log(torch.abs(grad_y))
    assert torch.allclose(ldj, expected_ldj, atol=1e-5)


def test_logit01_transform_roundtrip_and_jacobian():
    scale = 0.5
    t = Logit01Transform(scale=scale, delta=1e-4)
    x = torch.linspace(1e-3, 0.499, 50, requires_grad=True, dtype=torch.float32)
    y, ldj = t(x, training=False)
    x_rec = t.inverse(y)
    assert torch.allclose(x, x_rec, atol=1e-4)

    # Autograd Jacobian check
    grad_y = torch.autograd.grad(y.sum(), x)[0]
    expected_ldj = torch.log(torch.abs(grad_y))
    assert torch.allclose(ldj, expected_ldj, atol=1e-4)


def test_dequantize_transform_integer_recovery():
    t = DequantizeTransform(min_val=1.0)
    counts = torch.randint(low=1, high=80, size=(100,), dtype=torch.float32)
    # Forward in training mode (adds uniform noise in [0, 1))
    y, ldj = t(counts, training=True)
    rec = t.inverse(y)
    assert torch.equal(rec, counts)


def test_zero_inflated_logit_transform():
    t = ZeroInflatedLogitTransform(scale=0.5, delta=1e-4, noise_scale=1e-4)
    # Include exact zeros and non-zeros
    x = torch.tensor([0.0, 0.0, 0.1, 0.25, 0.45], dtype=torch.float32)
    y, _ = t(x, training=False)
    rec = t.inverse(y)
    assert rec[0].item() == 0.0
    assert rec[1].item() == 0.0
    assert torch.allclose(rec[2:], x[2:], atol=1e-3)


def test_standardize_transform():
    t = StandardizeTransform(dim=3)
    data = torch.randn(1000, 3) * 5.0 + 10.0
    t.fit(data)
    y, ldj = t(data, training=False)
    rec = t.inverse(y)
    assert torch.allclose(data, rec, atol=1e-5)
    assert torch.allclose(y.mean(dim=0), torch.zeros(3), atol=1e-2)
    assert torch.allclose(y.std(dim=0), torch.ones(3), atol=1e-2)


def test_full_pipeline_roundtrip():
    config = {
        "mass": {"type": "log", "eps": 1e-3},
        "mult": {"type": "dequantize", "min_val": 1.0},
        "zg": {"type": "zero_inflated_logit", "scale": 0.5, "delta": 1e-4, "noise_scale": 1e-4},
    }
    feature_names = ["mass", "mult", "zg"]
    pipe = Pipeline(feature_names, config)

    n_samples = 500
    torch.manual_seed(42)
    mass = torch.rand(n_samples) * 150.0 + 5.0
    mult = torch.randint(1, 50, (n_samples,)).float()
    zg = torch.cat([torch.zeros(50), torch.rand(n_samples - 50) * 0.4 + 0.1])

    data = torch.stack([mass, mult, zg], dim=-1)
    pipe.fit(data)

    # In eval mode (no noise), roundtrip should be very accurate
    z, ldj = pipe(data, training=False)
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()
    assert not torch.isnan(ldj).any()

    rec = pipe.inverse(z)
    assert not torch.isnan(rec).any()

    # Check mass
    assert torch.allclose(rec[:, 0], data[:, 0], atol=1e-3)
    # Check mult
    assert torch.equal(rec[:, 1], data[:, 1])
    # Check zg zeros
    assert (rec[:50, 2] == 0.0).all()
