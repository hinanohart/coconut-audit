"""Shared pytest fixtures for coconut-audit."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def skip_hf_download() -> bool:
    """True when CI is configured to skip HF Hub downloads (for offline runs)."""
    return os.environ.get("COCONUT_AUDIT_SKIP_HF_DOWNLOAD", "0") == "1"


@pytest.fixture
def tiny_random_seed() -> int:
    return 0xC0C0
