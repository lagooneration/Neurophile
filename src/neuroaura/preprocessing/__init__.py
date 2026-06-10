"""
neuroaura.preprocessing
========================
Signal processing pipelines.

Implemented
-----------
StandardPipeline        : MNE-based offline pipeline for normal-hearing EEG
CIArtifactPipeline      : Three-stage CI artifact rejection (Stage 1 complete)

Scaffold (see CONTRIBUTING.md)
------------------------------
ICAWrapper              : CI-aware ICA — src/neuroaura/preprocessing/ica.py
"""

from neuroaura.preprocessing.standard import StandardPipeline
from neuroaura.preprocessing.ci_artifact.pipeline import CIArtifactPipeline

__all__ = ["StandardPipeline", "CIArtifactPipeline"]
