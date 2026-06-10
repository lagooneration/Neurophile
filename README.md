# NeuroAuRA

**Neuro-Auditory Rehabilitation & Attention Platform**

An open-source EEG software ecosystem for auditory attention decoding, cochlear implant rehabilitation, and real-time neuroplasticity tracking.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What is NeuroAuRA?

NeuroAuRA is **not** a general-purpose BCI framework. It is a domain-specific platform for:

- **Auditory Attention Decoding (AAD):** Decode which audio stream a listener attends to from their EEG, using envelope-tracking correlations in the delta-theta band.
- **Cochlear Implant (CI) Rehabilitation:** Provide a closed-loop environment to test whether a CI patient's auditory cortex is successfully rewiring to degraded signals.
- **Longitudinal Plasticity Tracking:** Monitor N1/P2 evoked potential amplitudes and cortical tracking strength across rehabilitation sessions.
- **Federated Research:** Aggregate anonymized model updates across clinics to study neuroplasticity trends in tonal vs. non-tonal language speakers.

---

## Quick Start

```bash
pip install neuroaura

# Validate a BIDS-EEG dataset for AAD compliance
neuroaura validate /path/to/your/bids/dataset/

# Run an offline AAD evaluation on an existing dataset
neuroaura decode --dataset /path/to/bids/ --decoder linear --window 60

# Run the full CI rehabilitation pipeline (offline)
neuroaura decode --config configs/ci_rehab.yaml --dataset /path/to/bids/
```

---

## Supported Devices

| Device | Status | Driver | Channels | Notes |
|--------|--------|--------|----------|-------|
| OpenBCI Cyton | ✅ Implemented | `brainflow` | 8 | LSL-native, recommended for development |
| OpenBCI Ganglion | ✅ Implemented | `brainflow` | 4 | Bluetooth LE, screening use |
| Muse 2 / Muse S | 🔧 Scaffold | `muselsl` | 4–5 | Consumer, Tier 3 sync only |
| BrainProducts LiveAmp | 🔧 Scaffold | `brainvision-rda` | 32/64 | Research-grade reference device |
| g.tec g.USBamp | 🔧 Scaffold | `pygds` | 16/32 | CI-lab gold standard |
| Emotiv EPOC X | 🔧 Scaffold | `cortex` | 14 | Consumer, proprietary SDK |
| Any LSL-compatible | ✅ Generic | `pylsl` | any | Use `neuroaura.devices.lsl_generic` |

---

## Architecture Overview

```
neuroaura/
├── data/           Data standards: BIDS-EEG + HDF5 streaming
├── sync/           Temporal sync: TTL (Tier 1), LSL (Tier 2), Software (Tier 3)
├── stimulus/       Stimulus delivery: envelope extraction, paradigm scripting
├── devices/        EEG device drivers (LSL-based)
├── preprocessing/  Signal processing: standard pipeline + CI artifact rejection
├── decoding/       AAD decoders + parallel evaluation harness
├── federated/      Federated learning: edge training + server aggregation
├── visualization/  Real-time dashboard + session reports
└── cli/            `neuroaura` command-line interface
```

See [docs/architecture.md](docs/architecture.md) for the full specification.

---

## Synchronization Tiers

Auditory Attention Decoding requires < 3 ms audio-EEG temporal alignment. The platform defines three tiers:

| Tier | Method | Jitter | Status |
|------|--------|--------|--------|
| 1 | Hardware TTL trigger | < 0.5 ms | 🔧 Phase 2 |
| 2 | Lab Streaming Layer (LSL) | 1–3 ms | 🔧 Phase 2 |
| 3 | Software + calibration chirp | 5–15 ms | ✅ Phase 1 |

---

## CI Artifact Pipeline

Cochlear implant EEG artifacts are **not** removable with standard ICA/ASR. NeuroAuRA uses a three-stage pipeline:

| Stage | Method | Status |
|-------|--------|--------|
| 1 | Template subtraction (periodic artifact) | ✅ Implemented |
| 2 | Spatial filtering (CCA / LCMV / SSD) | 🔧 Scaffold — see [CONTRIBUTING.md](CONTRIBUTING.md) |
| 3 | Adaptive filtering (LMS / RLS) | 🔧 Scaffold — see [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Decoders

| Decoder | Type | Status | Reference |
|---------|------|--------|-----------|
| Linear stimulus reconstruction | Ridge regression | ✅ Implemented | Crosse et al. (2016) |
| CNN-based AAD | Deep learning | 🔧 Scaffold | Vandecappelle et al. (2021) |
| Subject-independent AAD | Transfer learning | 🔧 Scaffold | Ciccarelli et al. (2023) |

---

## Stimuli

All audio stimuli bundled with NeuroAuRA are CC-licensed. See [stimuli/manifest.yaml](stimuli/manifest.yaml) and [stimuli/LICENSE.md](stimuli/LICENSE.md) for per-file attribution.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The highest-value open contributions are:

1. **CI artifact Stage 2 & 3** (`src/neuroaura/preprocessing/ci_artifact/`)
2. **New EEG device drivers** (`src/neuroaura/devices/`)
3. **AAD decoder implementations** (`src/neuroaura/decoding/`)
4. **Real-time LSL pipeline** (`src/neuroaura/sync/`)

---

## Citation

If you use NeuroAuRA in a publication, please cite:

```bibtex
@software{neuroaura2025,
  title  = {NeuroAuRA: Neuro-Auditory Rehabilitation \& Attention Platform},
  year   = {2025},
  url    = {https://github.com/neuroaura/neuroaura},
  license = {Apache-2.0}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
