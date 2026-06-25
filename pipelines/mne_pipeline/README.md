# MNE Pipeline

A pure-Python EEG preprocessing and training pipeline with **no MATLAB or EEGLAB dependency**.

This is a parallel alternative to the EEGLAB-based pipeline in `scripts/eeglab_bridge.py`, which remains untouched and can be used whenever MATLAB is available.

---

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Core MNE preprocessing module (bandpass, bad channel detection, ICA) |
| `train_mne_pipeline.py` | End-to-end training script using the preprocessor above |

---

## Equivalent EEGLAB Operations

| EEGLAB Function | This Pipeline |
|---|---|
| `pop_eegfiltnew()` | `raw.filter(l_freq, h_freq)` |
| `clean_rawdata()` | `detect_and_interpolate_bad_channels()` |
| `pop_runica()` | `ICA(method="infomax", extended=True)` |
| `pop_iclabel()` | `mne_icalabel.label_components()` |

---

## Setup

Install the required extra package:
```powershell
pip install mne-icalabel
```

---

## Usage

**Single subject (to test the pipeline works):**
```powershell
python pipelines/mne_pipeline/train_mne_pipeline.py `
    --bids-root "F:\neurophile_data\ds003516" `
    --subject "001" `
    --device cuda `
    --epochs 5
```

**All 25 subjects (full global model training):**
```powershell
python pipelines/mne_pipeline/train_mne_pipeline.py `
    --bids-root "F:\neurophile_data\ds003516" `
    --subject "all" `
    --device cuda `
    --epochs 10
```

**Skip preprocessing (if data is already clean):**
```powershell
python pipelines/mne_pipeline/train_mne_pipeline.py `
    --bids-root "F:\neurophile_data\ds003516" `
    --subject "all" `
    --device cuda `
    --no-preprocess
```
