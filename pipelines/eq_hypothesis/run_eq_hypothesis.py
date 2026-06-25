"""
pipelines/eq_hypothesis/run_eq_hypothesis.py

Tests the hypothesis:
    "Audio passed through an EQ bandpass filter at specific frequency ranges
     produces a STRONGER Pearson correlation with the EEG Temporal Response
     Function (TRF) than broadband audio."

Pipeline:
    1. Load raw audio stimulus (.mat) from the BIDS stimuli/ folder
    2. Apply 6 EQ bandpass filters at different frequency ranges
    3. Extract the amplitude envelope for each filtered audio signal
    4. Load EEG for the subject (optionally ICA-cleaned)
    5. Compute the Pearson TRF correlation for each envelope vs EEG
    6. Plot a ranked bar chart of EQ bands by correlation strength

Usage:
    python pipelines/eq_hypothesis/run_eq_hypothesis.py \\
        --bids-root "F:\\neurophile_data\\ds003516" \\
        --subject "001"
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipelines.mne_pipeline.preprocess import clean_raw_eeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("eq_hypothesis")
mne.set_log_level("WARNING")

# ── EQ Band Definitions ───────────────────────────────────────────────────────
# Each entry: (label, low_hz, high_hz, description)
EQ_BANDS = [
    ("Broadband",    0.5,   8000.0, "Full spectrum (baseline)"),
    ("Sub-Bass",     20.0,  250.0,  "Prosody · Delta/Theta cortical following"),
    ("Bass",         250.0, 500.0,  "Fundamental pitch · N100 onset"),
    ("Low-Mid",      500.0, 1000.0, "Vowel formants F1 · strong cortical tracking"),
    ("Mid",          1000.0,4000.0, "Speech intelligibility · formants F2/F3"),
    ("High-Mid",     4000.0,8000.0, "Consonants · sibilants · onset only"),
]


def bandpass_audio(audio: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to mono audio."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.9999)
    
    if low >= high:
        return audio  # can't filter, return as-is
    
    b, a = signal.butter(4, [low, high], btype="band")
    return signal.filtfilt(b, a, audio).astype("float32")


def extract_envelope(audio: np.ndarray, fs: float, target_fs: float = 64.0) -> np.ndarray:
    """
    Extract the amplitude envelope of audio:
        1. Full-wave rectify (abs)
        2. Low-pass filter at 30 Hz to smooth
        3. Downsample to match the EEG sampling rate
    """
    # Rectify
    rectified = np.abs(audio)
    
    # Low-pass at 30 Hz
    nyq = fs / 2.0
    b, a = signal.butter(4, 30.0 / nyq, btype="low")
    smoothed = signal.filtfilt(b, a, rectified)
    
    # Downsample to EEG sampling rate
    n_samples_target = int(len(smoothed) * target_fs / fs)
    envelope = signal.resample(smoothed, n_samples_target).astype("float32")
    
    return envelope


def compute_trf_pearson(eeg_data: np.ndarray, envelope: np.ndarray,
                        lags_ms: tuple = (-200, 800), fs: float = 500.0) -> dict:
    """
    Compute the Temporal Response Function (TRF) by cross-correlating the
    audio envelope with each EEG channel at a range of time lags.
    
    Returns:
        dict with keys:
            'lags_ms'     : array of lag values in milliseconds
            'trf'         : (n_channels, n_lags) cross-correlation matrix
            'mean_trf'    : (n_lags,) mean TRF across channels
            'peak_lag_ms' : lag at which the peak absolute correlation occurs
            'peak_r'      : peak Pearson r value
            'mean_r'      : mean abs Pearson r across all lags
    """
    lag_min = int(lags_ms[0] * fs / 1000)
    lag_max = int(lags_ms[1] * fs / 1000)
    lags = np.arange(lag_min, lag_max + 1)
    lags_time = lags / fs * 1000  # convert to ms
    
    n_channels = eeg_data.shape[1]
    n_lags = len(lags)
    trf = np.zeros((n_channels, n_lags))
    
    # Align lengths
    min_len = min(eeg_data.shape[0], len(envelope))
    eeg_aligned = eeg_data[:min_len, :]
    env_aligned = envelope[:min_len]
    
    for lag_i, lag in enumerate(lags):
        if lag >= 0:
            eeg_lagged = eeg_aligned[lag:, :]
            env_lagged = env_aligned[:min_len - lag]
        else:
            abs_lag = abs(lag)
            eeg_lagged = eeg_aligned[:min_len - abs_lag, :]
            env_lagged = env_aligned[abs_lag:]
        
        trim = min(len(eeg_lagged), len(env_lagged))
        eeg_lagged = eeg_lagged[:trim, :]
        env_lagged = env_lagged[:trim]
        
        for ch in range(n_channels):
            r, _ = pearsonr(eeg_lagged[:, ch], env_lagged)
            trf[ch, lag_i] = r if not np.isnan(r) else 0.0
    
    mean_trf = trf.mean(axis=0)
    peak_idx = np.argmax(np.abs(mean_trf))
    
    return {
        "lags_ms": lags_time,
        "trf": trf,
        "mean_trf": mean_trf,
        "peak_lag_ms": lags_time[peak_idx],
        "peak_r": mean_trf[peak_idx],
        "mean_r": np.abs(mean_trf).mean(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test EQ-filtered audio TRF correlation hypothesis"
    )
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--subject", type=str, default="001")
    parser.add_argument(
        "--no-ica", action="store_true",
        help="Skip ICA preprocessing (faster but noisier EEG)"
    )
    parser.add_argument(
        "--audio-file", type=Path, default=None,
        help="Override audio file (default: first .mat in stimuli/)"
    )
    args = parser.parse_args()

    # ── 1. Load EEG ───────────────────────────────────────────────────────────
    eeg_dir = args.bids_root / f"sub-{args.subject}" / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    if not set_files:
        logger.error("No .set file found in %s", eeg_dir)
        return

    logger.info("Loading EEG: %s", set_files[0].name)
    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
    raw.pick_types(eeg=True)
    eeg_fs = raw.info["sfreq"]

    if not args.no_ica:
        logger.info("Applying MNE preprocessing pipeline (bandpass + ICA)...")
        raw = clean_raw_eeg(raw, bandpass_l=1.0, bandpass_h=40.0, n_ica_components=20)
    
    eeg_data = raw.get_data().T.astype("float32")  # (Time, Channels)

    # ── 2. Load Audio ─────────────────────────────────────────────────────────
    audio_file = args.audio_file
    if audio_file is None:
        stim_dir = args.bids_root / "stimuli"
        # Prefer full-signal audio (sig1_1.mat, sig2_1.mat etc)
        audio_files = sorted(stim_dir.glob("sig*.mat"))
        if not audio_files:
            audio_files = sorted(stim_dir.glob("*.mat"))
        audio_file = audio_files[0] if audio_files else None

    if audio_file is None or not audio_file.exists():
        logger.error("No audio .mat file found in %s/stimuli/", args.bids_root)
        return

    logger.info("Loading audio: %s", audio_file.name)
    mat = sio.loadmat(str(audio_file))
    audio_key = [k for k in mat if not k.startswith("__")][0]
    raw_audio = mat[audio_key].astype("float32").flatten()

    # Assume audio was recorded at 44100 Hz (standard) unless we can detect it
    audio_fs = 44100.0
    logger.info("Audio samples: %d at %.0f Hz (%.1f sec)",
                len(raw_audio), audio_fs, len(raw_audio) / audio_fs)

    # ── 3. Compute TRF for each EQ band ──────────────────────────────────────
    results = []
    envelopes = {}

    for band_name, low_hz, high_hz, description in EQ_BANDS:
        logger.info("Processing EQ band: %s (%.0f–%.0f Hz)...", band_name, low_hz, high_hz)

        # Filter the audio
        filtered = bandpass_audio(raw_audio, audio_fs, low_hz, high_hz)

        # Extract amplitude envelope, downsampled to EEG rate
        envelope = extract_envelope(filtered, audio_fs, target_fs=eeg_fs)
        envelopes[band_name] = envelope

        # Compute TRF correlation
        trf_result = compute_trf_pearson(eeg_data, envelope, lags_ms=(-200, 800), fs=eeg_fs)
        results.append({
            "band": band_name,
            "low_hz": low_hz,
            "high_hz": high_hz,
            "description": description,
            **trf_result,
        })
        logger.info("  → Peak r=%.4f at %.0f ms | Mean r=%.4f",
                    trf_result["peak_r"], trf_result["peak_lag_ms"], trf_result["mean_r"])

    # ── 4. Rank and Print Results ─────────────────────────────────────────────
    results.sort(key=lambda x: abs(x["peak_r"]), reverse=True)

    print("\n" + "=" * 65)
    print(f"EQ HYPOTHESIS RESULTS — Subject {args.subject}")
    print("=" * 65)
    print(f"{'Rank':<5} {'Band':<12} {'Freq Range':<18} {'Peak r':>8} {'Peak Lag':>10} {'Mean r':>8}")
    print("-" * 65)
    for i, r in enumerate(results):
        freq_str = f"{r['low_hz']:.0f}–{r['high_hz']:.0f} Hz"
        winner = " ← STRONGEST" if i == 0 else ""
        print(f"{i+1:<5} {r['band']:<12} {freq_str:<18} {r['peak_r']:>+8.4f} {r['peak_lag_ms']:>9.0f}ms {r['mean_r']:>8.4f}{winner}")
    print("=" * 65)

    # ── 5. Generate Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"EQ-Filtered Audio TRF Hypothesis — Subject {args.subject}\n"
        f"Audio: {audio_file.name}",
        fontsize=13, fontweight="bold"
    )

    # Plot A: TRF curves for each EQ band
    ax_trf = axes[0, 0]
    colors = ["gray", "navy", "royalblue", "green", "orange", "red"]
    for res, color in zip(results, colors):
        ax_trf.plot(res["lags_ms"], res["mean_trf"],
                    label=f"{res['band']} ({res['low_hz']:.0f}–{res['high_hz']:.0f} Hz)",
                    color=color, linewidth=1.5 if res["band"] == "Broadband" else 1.0,
                    alpha=0.9)
    ax_trf.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_trf.axvline(0, color="black", linewidth=0.5, linestyle=":")
    # Mark known ERP latencies
    for lat, label, col in [(100, "N100", "red"), (200, "P200", "blue"), (300, "P300", "green")]:
        ax_trf.axvline(lat, color=col, linestyle="--", alpha=0.4)
        ax_trf.text(lat + 5, ax_trf.get_ylim()[0] * 0.8, label, color=col, fontsize=8)
    ax_trf.set_xlabel("Time lag (ms)")
    ax_trf.set_ylabel("Mean Pearson r (across channels)")
    ax_trf.set_title("TRF Correlation by EQ Band")
    ax_trf.legend(fontsize=8)

    # Plot B: Bar chart of peak correlation by band
    ax_bar = axes[0, 1]
    band_names = [r["band"] for r in results]
    peak_rs = [abs(r["peak_r"]) for r in results]
    bar_colors = ["gold" if i == 0 else "steelblue" for i in range(len(results))]
    bars = ax_bar.bar(band_names, peak_rs, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax_bar.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
    ax_bar.set_xlabel("EQ Band")
    ax_bar.set_ylabel("|Peak Pearson r|")
    ax_bar.set_title("Hypothesis Test: Which EQ Band Maximizes Correlation?")
    ax_bar.tick_params(axis="x", rotation=15)

    # Plot C: Audio envelopes for each band
    ax_env = axes[1, 0]
    t_audio = np.arange(min(2000, min(len(v) for v in envelopes.values()))) / eeg_fs
    for (band_name, _, _, _), color in zip(EQ_BANDS, colors):
        env = envelopes[band_name]
        ax_env.plot(t_audio, env[:len(t_audio)], label=band_name, color=color, alpha=0.7, linewidth=0.8)
    ax_env.set_xlabel("Time (s)")
    ax_env.set_ylabel("Amplitude")
    ax_env.set_title("Audio Envelopes by EQ Band (first 2000 samples)")
    ax_env.legend(fontsize=8)

    # Plot D: Summary heatmap — mean abs r across lag × band
    ax_heat = axes[1, 1]
    trf_matrix = np.array([r["mean_trf"] for r in results])
    lag_axis = results[0]["lags_ms"]
    im = ax_heat.imshow(
        np.abs(trf_matrix),
        aspect="auto",
        origin="upper",
        extent=[lag_axis[0], lag_axis[-1], len(results) - 0.5, -0.5],
        cmap="hot",
    )
    plt.colorbar(im, ax=ax_heat, label="|Pearson r|")
    ax_heat.set_yticks(range(len(results)))
    ax_heat.set_yticklabels([r["band"] for r in results], fontsize=9)
    ax_heat.set_xlabel("Time lag (ms)")
    ax_heat.set_title("TRF Heatmap: |r| across Lag × EQ Band")
    for lat, label, col in [(100, "N100", "white"), (200, "P200", "cyan"), (300, "P300", "lime")]:
        ax_heat.axvline(lat, color=col, linestyle="--", alpha=0.6, linewidth=1.2)
        ax_heat.text(lat + 5, 0.1, label, color=col, fontsize=7)

    plt.tight_layout()
    
    # Save the figure
    out_path = Path("checkpoints") / f"eq_hypothesis_sub{args.subject}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Figure saved to %s", out_path)

    print(f"\nFigure saved → {out_path}")
    print("Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
