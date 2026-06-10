"""
neuroaura.preprocessing.ci_artifact.pipeline
=============================================
Orchestrator for the three-stage CI artifact rejection pipeline.

Runs whichever stages are enabled in the config. Disabled stages pass
the signal through unchanged, allowing partial pipelines during development.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from neuroaura.preprocessing.ci_artifact.template_subtraction import (
    CITemplateSubtraction,
    TemplateSubtractionConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CIArtifactConfig:
    """Master configuration for the full CI artifact pipeline."""

    stage1: TemplateSubtractionConfig = None  # type: ignore[assignment]
    stage2_enabled: bool = False  # set True once spatial_filter.py is implemented
    stage3_enabled: bool = False  # set True once adaptive_filter.py is implemented

    def __post_init__(self) -> None:
        if self.stage1 is None:
            self.stage1 = TemplateSubtractionConfig()


class CIArtifactPipeline:
    """Three-stage CI artifact rejection pipeline.

    Only Stage 1 is currently implemented. Stages 2 and 3 are no-ops
    until their respective modules are contributed.

    Parameters
    ----------
    fs : int
        EEG sampling rate in Hz.
    config : CIArtifactConfig
        Pipeline configuration. Defaults: Stage 1 enabled, 2+3 disabled.

    Examples
    --------
    >>> pipeline = CIArtifactPipeline(fs=1000)
    >>> clean_eeg = pipeline.run(raw_eeg)  # (n_samples, n_channels)
    >>> quality = pipeline.quality_report(raw_eeg, clean_eeg)
    """

    def __init__(
        self,
        fs: int,
        config: CIArtifactConfig | None = None,
    ) -> None:
        self.fs = fs
        self.config = config or CIArtifactConfig()
        self._stage1 = CITemplateSubtraction(fs=fs, config=self.config.stage1)

    def run(self, eeg: np.ndarray) -> np.ndarray:
        """Apply all enabled stages in sequence.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
            Raw EEG with CI artifact.

        Returns
        -------
        clean_eeg : np.ndarray, shape (n_samples, n_channels)
        """
        logger.info("CI Artifact Pipeline: input shape %s", eeg.shape)

        # ── Stage 1: Template Subtraction ─────────────────────────────────────
        eeg = self._stage1.fit_transform(eeg)
        detected_rate = self._stage1.detected_rate_pps
        if detected_rate:
            logger.info("  Stage 1 done. CI rate ≈ %.1f pps.", detected_rate)

        # ── Stage 2: Spatial Filter ───────────────────────────────────────────
        if self.config.stage2_enabled:
            logger.warning(
                "Stage 2 (spatial filter) is enabled but not yet implemented. "
                "See src/neuroaura/preprocessing/ci_artifact/spatial_filter.py"
            )
        else:
            logger.debug("  Stage 2 (spatial filter): disabled — skipping.")

        # ── Stage 3: Adaptive Filter ──────────────────────────────────────────
        if self.config.stage3_enabled:
            logger.warning(
                "Stage 3 (adaptive filter) is enabled but not yet implemented. "
                "See src/neuroaura/preprocessing/ci_artifact/adaptive_filter.py"
            )
        else:
            logger.debug("  Stage 3 (adaptive filter): disabled — skipping.")

        logger.info("CI Artifact Pipeline: output shape %s", eeg.shape)
        return eeg

    def quality_report(
        self,
        raw_eeg: np.ndarray,
        clean_eeg: np.ndarray,
    ) -> dict[str, float]:
        """Compute post-cleaning quality metrics.

        Parameters
        ----------
        raw_eeg : np.ndarray
            Original EEG before pipeline.
        clean_eeg : np.ndarray
            EEG after pipeline.

        Returns
        -------
        metrics : dict
            Keys: snr_improvement_db, neural_band_power_ratio,
                  artifact_residual_correlation, topographic_asymmetry_index.
        """
        from neuroaura.preprocessing.ci_artifact.quality import CIArtifactQuality
        return CIArtifactQuality.compute(raw_eeg, clean_eeg, self.fs)
