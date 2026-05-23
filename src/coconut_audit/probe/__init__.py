"""Latent-activation probes for Coconut / Quiet-STaR / TTC families."""

from coconut_audit.probe.client import HFInferenceClient
from coconut_audit.probe.extractor import LatentExtractor
from coconut_audit.probe.registry import HookRegistry, hook_site_for

__all__ = [
    "HFInferenceClient",
    "HookRegistry",
    "LatentExtractor",
    "hook_site_for",
]
