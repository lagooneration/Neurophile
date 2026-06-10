"""
neuroaura.preprocessing.ci_artifact
=====================================
Three-stage cochlear implant artifact rejection pipeline.

Stage   Module                  Status
------  ----------------------  ----------------------------
1       template_subtraction    ✅ Implemented
2       spatial_filter          🔧 Scaffold — contribute!
3       adaptive_filter         🔧 Scaffold — contribute!

The pipeline orchestrator (pipeline.py) runs whichever stages are enabled
in the YAML config. Disabled stages pass data through unchanged.
"""

from neuroaura.preprocessing.ci_artifact.pipeline import CIArtifactPipeline
from neuroaura.preprocessing.ci_artifact.quality import CIArtifactQuality

__all__ = ["CIArtifactPipeline", "CIArtifactQuality"]
