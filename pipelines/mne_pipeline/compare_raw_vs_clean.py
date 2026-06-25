"""
pipelines/mne_pipeline/compare_raw_vs_clean.py

Visualizes the before/after effect of MNE preprocessing for a specific subject.
Shows side-by-side comparisons of:
  1. Raw EEG scroll vs Cleaned EEG scroll
  2. PSD before and after (frequency domain)
  3. ICA component maps (which were removed)
  4. Overlay butterfly plot of a specific channel

Usage:
    python pipelines/mne_pipeline/compare_raw_vs_clean.py \\
        --bids-root "F:\\neurophile_data\\ds003516" \\
        --subject "001"
"""

import argparse
import logging
import sys
from pathlib import Path

import mne
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipelines.mne_pipeline.preprocess import (
    detect_and_interpolate_bad_channels,
    run_ica_artifact_rejection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("compare_raw_vs_clean")

mne.set_log_level("WARNING")  # suppress MNE's verbose output for cleaner prints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--subject", type=str, default="001")
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Specific channel name to overlay (e.g. 'E1'). Default: auto-pick first EEG channel.",
    )
    args = parser.parse_args()

    # ── 1. Load raw data ───────────────────────────────────────────────────────
    eeg_dir = args.bids_root / f"sub-{args.subject}" / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    if not set_files:
        print(f"No .set file found in {eeg_dir}")
        return

    logger.info("Loading raw EEG: %s", set_files[0])
    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
    raw.pick_types(eeg=True)

    # Keep an untouched copy for comparison
    raw_original = raw.copy()

    # ── 2. Run preprocessing pipeline ────────────────────────────────────────
    logger.info("Applying bandpass filter 1–40 Hz...")
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)

    logger.info("Detecting bad channels...")
    raw_interp = detect_and_interpolate_bad_channels(raw_filtered.copy())
    bad_channels = raw_filtered.info["bads"]

    logger.info("Running ICA artifact rejection...")

    # Fit ICA on the filtered data and keep a reference before applying
    ica = mne.preprocessing.ICA(
        n_components=20, method="infomax",
        fit_params={"extended": True}, random_state=42, max_iter="auto"
    )
    ica.fit(raw_interp)

    # Auto-detect EOG components as fallback
    try:
        from mne_icalabel import label_components
        ic_labels = label_components(raw_interp, ica, method="iclabel")
        exclude_idx = [
            i for i, (label, prob) in enumerate(
                zip(ic_labels["labels"], ic_labels["y_pred_proba"])
            )
            if label not in ("brain", "other") and prob.max() > 0.70
        ]
    except ImportError:
        exclude_idx, _ = ica.find_bads_eog(raw_interp, threshold=3.0)

    ica.exclude = exclude_idx
    raw_clean = ica.apply(raw_interp.copy())

    logger.info("Rejected components: %s", exclude_idx)

    # ── 3. Plot 1: PSD Comparison (Before vs After) ───────────────────────────
    print("\n[1/4] Generating Power Spectral Density comparison...")
    fig_psd, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_psd.suptitle(f"Subject {args.subject} — PSD: Raw vs Cleaned", fontsize=14, fontweight="bold")

    raw_original.compute_psd(fmax=50).plot(axes=axes[0], show=False)
    axes[0].set_title("RAW (before cleaning)", color="red")

    raw_clean.compute_psd(fmax=50).plot(axes=axes[1], show=False)
    axes[1].set_title("CLEAN (after ICA + bad channel rejection)", color="green")

    # ── 4. Plot 2: Single Channel Overlay ────────────────────────────────────
    print("[2/4] Generating per-channel overlay comparison...")
    channel = args.channel or raw_original.ch_names[0]
    ch_idx_orig = raw_original.ch_names.index(channel)
    ch_idx_clean = raw_clean.ch_names.index(channel)

    t_start = int(raw_original.info["sfreq"] * 10)   # from 10s
    t_end   = int(raw_original.info["sfreq"] * 20)   # to 20s
    times   = raw_original.times[t_start:t_end]

    data_raw   = raw_original.get_data()[ch_idx_orig,  t_start:t_end] * 1e6   # µV
    data_clean = raw_clean.get_data()[ch_idx_clean,    t_start:t_end] * 1e6

    fig_ch, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, data_raw,   color="red",   alpha=0.6, linewidth=0.8, label="Raw")
    ax.plot(times, data_clean, color="green", alpha=0.9, linewidth=0.8, label="Cleaned")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Subject {args.subject} — Channel {channel}: Raw (red) vs Cleaned (green)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    # Shade regions of large raw amplitude (likely eye-blink zones)
    blink_threshold = np.percentile(np.abs(data_raw), 95)
    blink_mask = np.abs(data_raw) > blink_threshold
    ax.fill_between(
        times, data_raw.min(), data_raw.max(),
        where=blink_mask, alpha=0.15, color="red", label="Possible blink region"
    )

    # ── 5. Plot 3: ICA Component Topomaps ────────────────────────────────────
    print("[3/4] Generating ICA component maps (rejected components shown in red)...")
    fig_ica = ica.plot_components(show=False)
    if isinstance(fig_ica, list):
        fig_ica = fig_ica[0]
    fig_ica.suptitle(
        f"Subject {args.subject} — ICA Components "
        f"(Rejected: {exclude_idx if exclude_idx else 'none detected'})",
        fontsize=12,
        fontweight="bold",
    )

    # ── 6. Plot 4: Bad channel summary ───────────────────────────────────────
    print("[4/4] Generating bad channel summary table...")
    fig_txt, ax_txt = plt.subplots(figsize=(8, 3))
    ax_txt.axis("off")
    summary_lines = [
        f"Subject: {args.subject}",
        f"Total EEG Channels: {len(raw_original.ch_names)}",
        f"Bad Channels Interpolated: {bad_channels if bad_channels else 'None'}",
        f"ICA Components Decomposed: 20",
        f"Artifact Components Removed: {exclude_idx if exclude_idx else 'None'}",
        f"Sampling Rate: {raw_original.info['sfreq']} Hz",
        f"Recording Duration: {raw_original.times[-1]:.1f}s",
    ]
    ax_txt.text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=ax_txt.transAxes,
        fontsize=12, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    fig_txt.suptitle(f"Subject {args.subject} — Preprocessing Report", fontweight="bold")

    print("\nDisplaying all plots. Close each window to see the next one.")
    plt.show()


if __name__ == "__main__":
    main()
