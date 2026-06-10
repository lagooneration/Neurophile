"""
neuroaura.stimulus
==================
Stimulus delivery, envelope extraction, and paradigm scripting.

Implemented
-----------
StimulusManifest    : CC-licensed audio registry with SHA-256 checksums
EnvelopeExtractor   : Gammatone filterbank → Hilbert envelope

Scaffold (see CONTRIBUTING.md)
------------------------------
StimulusEngine      : Low-latency audio playback (PortAudio/sounddevice) — engine.py
ParadigmRunner      : YAML paradigm file parser and trial runner — paradigm.py
"""

from neuroaura.stimulus.manifest import StimulusManifest
from neuroaura.stimulus.envelope import EnvelopeExtractor

__all__ = ["StimulusManifest", "EnvelopeExtractor"]
