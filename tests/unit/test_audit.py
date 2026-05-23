"""Unit tests for coconut_audit.audit (synthetic tensors; no model needed)."""

from __future__ import annotations

import pytest
import torch

from coconut_audit.audit import (
    AuditVerdict,
    DriftScorer,
    ShortcutDetector,
    SteeringProbe,
    Verdict,
    aggregate_verdict,
    label_for_metric,
    latent_drift_score,
)


def test_steering_probe_detects_higher_latent_sensitivity() -> None:
    g = torch.Generator().manual_seed(0xC0C0)
    clean = torch.randn(2, 6, 8, generator=g)
    # Inject 10× more perturbation on latent (mask==0) than CoT (mask==1).
    noise = torch.randn_like(clean)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    noise[mask == 0] *= 10
    probe = SteeringProbe()
    out = probe.run(clean, clean + noise, mask)
    assert out["latent_sensitivity"] > out["cot_sensitivity"]
    assert out["ratio"] > 5.0


def test_steering_probe_shape_validation() -> None:
    probe = SteeringProbe()
    with pytest.raises(ValueError, match="shape mismatch"):
        probe.run(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="token_kind_mask"):
        probe.run(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 3))


def test_shortcut_detector_gap() -> None:
    id_correct = torch.tensor([1, 1, 1, 1, 0])
    ood_correct = torch.tensor([1, 0, 0, 0, 0])
    out = ShortcutDetector().run(id_correct, ood_correct)
    assert out["id_acc"] == pytest.approx(0.8)
    assert out["ood_acc"] == pytest.approx(0.2)
    assert out["gap"] == pytest.approx(0.6)


def test_shortcut_detector_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1D"):
        ShortcutDetector().run(torch.zeros(2, 3), torch.zeros(3))


def test_drift_scorer_zero_on_identical_distributions() -> None:
    a = torch.randn(64, 32)
    assert DriftScorer(method="wasserstein").score(a, a) == pytest.approx(0.0, abs=1e-6)
    assert DriftScorer(method="l2_mean").score(a, a) == pytest.approx(0.0, abs=1e-6)


def test_drift_scorer_grows_with_shift() -> None:
    g = torch.Generator().manual_seed(0xC0C0)
    a = torch.randn(128, 16, generator=g)
    b_small = a + 0.1 * torch.randn(128, 16, generator=g)
    b_big = a + 1.0 * torch.randn(128, 16, generator=g)
    d_small = DriftScorer().score(a, b_small)
    d_big = DriftScorer().score(a, b_big)
    assert d_big > d_small


def test_drift_scorer_feature_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="feature-dim mismatch"):
        DriftScorer().score(torch.zeros(8, 16), torch.zeros(8, 32))


def test_drift_scorer_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown drift method"):
        DriftScorer(method="nope").score(torch.zeros(2, 2), torch.zeros(2, 2))


def test_latent_drift_score_alias() -> None:
    a = torch.zeros(4, 4)
    assert latent_drift_score(a, a) == pytest.approx(0.0)


def test_label_for_metric_higher_is_worse() -> None:
    assert label_for_metric(0.05, warn_at=0.1, fail_at=0.3) == AuditVerdict.PASS
    assert label_for_metric(0.2, warn_at=0.1, fail_at=0.3) == AuditVerdict.WARN
    assert label_for_metric(0.5, warn_at=0.1, fail_at=0.3) == AuditVerdict.FAIL


def test_label_for_metric_lower_is_worse() -> None:
    assert (
        label_for_metric(0.9, warn_at=0.5, fail_at=0.3, higher_is_worse=False) == AuditVerdict.PASS
    )
    assert (
        label_for_metric(0.5, warn_at=0.5, fail_at=0.3, higher_is_worse=False) == AuditVerdict.WARN
    )
    assert (
        label_for_metric(0.2, warn_at=0.5, fail_at=0.3, higher_is_worse=False) == AuditVerdict.FAIL
    )


def test_aggregate_verdict_priority() -> None:
    pass_v = Verdict(label=AuditVerdict.PASS, metric_name="m", metric_value=0.0, threshold=0.1)
    warn_v = Verdict(label=AuditVerdict.WARN, metric_name="m", metric_value=0.2, threshold=0.1)
    fail_v = Verdict(label=AuditVerdict.FAIL, metric_name="m", metric_value=0.5, threshold=0.3)
    assert aggregate_verdict([pass_v, pass_v]) == AuditVerdict.PASS
    assert aggregate_verdict([pass_v, warn_v]) == AuditVerdict.WARN
    assert aggregate_verdict([pass_v, warn_v, fail_v]) == AuditVerdict.FAIL
    assert aggregate_verdict([]) == AuditVerdict.NO_VERDICT
