"""Smoke tests: import + version sanity."""

from __future__ import annotations


def test_package_importable() -> None:
    import coconut_audit

    assert coconut_audit.__version__ == "0.1.0"


def test_submodule_imports() -> None:
    from coconut_audit import audit, core, mcp, probe, reports, sae

    assert audit is not None
    assert core is not None
    assert mcp is not None
    assert probe is not None
    assert reports is not None
    assert sae is not None
