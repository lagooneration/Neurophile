# EQ Hypothesis Pipeline

Tests the research hypothesis: 
> *"Audio passed through an EQ bandpass filter at specific frequency ranges produces a STRONGER Pearson correlation with the EEG Temporal Response Function (TRF) than broadband audio."*

## How It Works

```
Raw Audio (.mat)
      │
      ├── Broadband (0.5–8000 Hz) ──► Envelope ──► TRF r vs EEG
      ├── Sub-Bass  (20–250 Hz)   ──► Envelope ──► TRF r vs EEG
      ├── Bass      (250–500 Hz)  ──► Envelope ──► TRF r vs EEG
      ├── Low-Mid   (500–1000 Hz) ──► Envelope ──► TRF r vs EEG
      ├── Mid       (1–4 kHz)     ──► Envelope ──► TRF r vs EEG
      └── High-Mid  (4–8 kHz)    ──► Envelope ──► TRF r vs EEG
                                             │
                                             ▼
                                  Ranked bar chart + TRF plot
```

## Run

```powershell
# Fast (no ICA, for quick test):
python pipelines/eq_hypothesis/run_eq_hypothesis.py `
    --bids-root "F:\neurophile_data\ds003516" `
    --subject "001" `
    --no-ica

# Full (with ICA cleaned EEG, for accurate result):
python pipelines/eq_hypothesis/run_eq_hypothesis.py `
    --bids-root "F:\neurophile_data\ds003516" `
    --subject "001"
```

## Output

Four-panel figure saved to `checkpoints/eq_hypothesis_sub001.png`:
1. TRF correlation curves for all 6 EQ bands (with N100/P200/P300 markers)
2. Bar chart ranked by strongest correlation
3. Audio envelopes for each EQ band
4. Heatmap of |r| across all lags and bands
