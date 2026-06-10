"""
neuroaura.decoding
==================
Auditory Attention Decoding (AAD) models and evaluation harness.

Implemented
-----------
LinearDecoder           : Ridge-regression stimulus reconstruction (Crosse 2016)
AADEvaluator            : Parallel evaluation harness (decoder-agnostic)

Scaffold (see CONTRIBUTING.md)
------------------------------
CNNDecoder              : CNN-based AAD — cnn_decoder.py
PlasticityTracker       : Longitudinal N1/P2 model — plasticity.py
"""

from neuroaura.decoding.base import BaseDecoder
from neuroaura.decoding.linear_decoder import LinearDecoder
from neuroaura.decoding.aad_evaluation import AADEvaluator

__all__ = ["BaseDecoder", "LinearDecoder", "AADEvaluator"]
