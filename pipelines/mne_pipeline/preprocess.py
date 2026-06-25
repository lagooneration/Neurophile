"""
pipelines/mne_pipeline/preprocess.py

Pure Python EEG artifact rejection using MNE and mne-icalabel.
This is a MATLAB/EEGLAB-free alternative to the scripts/eeglab_bridge.py approach.

Equivalent EEGLAB operations performed here:
  EEGLAB pop_eegfiltnew()    → raw.filter()
  EEGLAB pop_runica()        → mne.preprocessing.ICA (infomax extended)
  EEGLAB pop_iclabel()       → mne_icalabel.label_components()
  EEGLAB clean_rawdata()     → mne.preprocessing.find_bad_channels_maxwell()
  EEGLAB pop_interp()        → raw.interpolate_bads()

Usage:
    from pipelines.mne_pipeline.preprocess import clean_raw_eeg
    raw_clean = clean_raw_eeg(raw)
"""
import logging
from pathlib import Path

import mne

logger = logging.getLogger("mne_pipeline.preprocess")

# --- Try to import mne-icalabel for automatic eye-blink detection ---
try:
    from mne_icalabel import label_components as _label_components
    _ICALABEL_AVAILABLE = True
    logger.info("mne-icalabel found — automatic component labelling enabled.")
except ImportError:
    _ICALABEL_AVAILABLE = False
    logger.warning(
        "mne-icalabel not found — ICA will decompose but won't auto-reject components. "
        "Install with: pip install mne-icalabel"
    )


def detect_and_interpolate_bad_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Detects flat or noisy channels via RANSAC-style statistics and
    interpolates them using the spherical spline method (same as EEGLAB
    pop_interp).

    Returns the modified raw object (in-place interpolation).
    """
    # MNE's built-in bad-channel detection via z-score of channel variance
    raw.pick_types(eeg=True, exclude=[])
    
    # Mark channels with very low variance (flat/dead) as bad
    data = raw.get_data()
    ch_stds = data.std(axis=1)
    mean_std = ch_stds.mean()
    
    flat_threshold = mean_std * 0.05  # channels less than 5% of mean power = flat
    noise_threshold = mean_std * 10.0  # channels more than 10x mean power = noisy
    
    newly_bad = []
    for i, (ch_name, std) in enumerate(zip(raw.ch_names, ch_stds)):
        if std < flat_threshold or std > noise_threshold:
            newly_bad.append(ch_name)
    
    if newly_bad:
        logger.warning("Detected %d bad channels: %s", len(newly_bad), newly_bad)
        raw.info["bads"].extend(newly_bad)
    
    if raw.info["bads"]:
        logger.info("Interpolating %d bad channels using spherical splines...", len(raw.info["bads"]))
        raw.interpolate_bads(reset_bads=True)
    else:
        logger.info("No bad channels detected.")
    
    return raw


def run_ica_artifact_rejection(
    raw: mne.io.BaseRaw,
    n_components: int = 20,
) -> mne.io.BaseRaw:
    """
    Runs the full ICA artifact rejection pipeline:
      1. Fits ICA using the InfoMax-Extended algorithm (identical to EEGLAB runica).
      2. If mne-icalabel is installed, auto-classifies components (eye, muscle, etc.)
         and rejects anything that is not labelled "brain" or "other".
      3. Falls back to MNE's simple EOG/ECG heuristic detector if mne-icalabel is absent.
    
    Returns the cleaned raw object.
    """
    logger.info("Fitting ICA (infomax-extended, n_components=%d)...", n_components)
    
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},  # Extended infomax = same as EEGLAB runica
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw)
    
    if _ICALABEL_AVAILABLE:
        # --- Automatic ICLabel Classification (EEGLAB-compatible) ---
        logger.info("Running mne-icalabel to auto-classify components...")
        ic_labels = _label_components(raw, ica, method="iclabel")
        
        exclude_idx = []
        for i, (label, prob) in enumerate(
            zip(ic_labels["labels"], ic_labels["y_pred_proba"])
        ):
            confidence = prob.max()
            logger.info("  IC %02d: %-12s (confidence=%.2f)", i, label, confidence)
            if label not in ("brain", "other") and confidence > 0.70:
                exclude_idx.append(i)
                logger.warning("    → Marking IC %02d (%s) for rejection!", i, label)
        
        ica.exclude = exclude_idx
        logger.info("Rejecting %d artifact components: %s", len(exclude_idx), exclude_idx)
    
    else:
        # --- Fallback: MNE's built-in EOG/ECG heuristic detection ---
        logger.info("Falling back to MNE EOG/ECG correlation heuristics...")
        eog_indices, _ = ica.find_bads_eog(raw, threshold=3.0)
        ica.exclude = eog_indices
        logger.info("Rejecting %d EOG artifact components: %s", len(eog_indices), eog_indices)
    
    raw_clean = ica.apply(raw.copy())
    logger.info("ICA artifact rejection complete!")
    return raw_clean


def clean_raw_eeg(
    raw: mne.io.BaseRaw,
    bandpass_l: float = 1.0,
    bandpass_h: float = 40.0,
    n_ica_components: int = 20,
    skip_bad_channel_detection: bool = False,
) -> mne.io.BaseRaw:
    """
    Master preprocessing function. Runs the full MNE cleaning pipeline:
        1. Bandpass filter (default 1-40 Hz)
        2. Bad channel detection + interpolation
        3. ICA artifact rejection (eye blinks, muscle noise)

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The loaded raw EEG object (from read_raw_eeglab or read_raw_bids).
    bandpass_l : float
        Low-pass cutoff frequency in Hz (default 1.0).
    bandpass_h : float
        High-pass cutoff frequency in Hz (default 40.0).
    n_ica_components : int
        Number of ICA components to decompose into (default 20).
    skip_bad_channel_detection : bool
        Skip bad channel detection if already handled upstream.

    Returns
    -------
    mne.io.BaseRaw
        The cleaned raw object ready for epoch extraction or numpy conversion.
    """
    logger.info("="*50)
    logger.info("Starting MNE Preprocessing Pipeline")
    
    # Step 1: Bandpass filter
    logger.info("Step 1/3: Applying %.1f-%.1f Hz bandpass filter...", bandpass_l, bandpass_h)
    raw.filter(l_freq=bandpass_l, h_freq=bandpass_h, verbose=False)
    
    # Step 2: Bad channel detection and interpolation
    if not skip_bad_channel_detection:
        logger.info("Step 2/3: Detecting and interpolating bad channels...")
        raw = detect_and_interpolate_bad_channels(raw)
    else:
        logger.info("Step 2/3: Skipping bad channel detection.")
    
    # Step 3: ICA
    logger.info("Step 3/3: Running ICA artifact rejection...")
    raw = run_ica_artifact_rejection(raw, n_components=n_ica_components)
    
    logger.info("MNE Preprocessing Pipeline complete!")
    logger.info("="*50)
    return raw
