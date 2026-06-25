"""
pipelines/eq_hypothesis/run_harmonizer_hypothesis.py

Tests the hypothesis:
    "Adding artificial overtones (harmonics) to a speech audio stimulus via a
     digital harmonizer will increase the Pearson TRF correlation with the EEG
     signal, because the auditory cortex has dedicated harmonic template circuits
     that produce stronger cortical entrainment to harmonically rich sounds."

Harmonizer Implementation:
    The harmonizer generates pitch-shifted copies of the dry audio at:
      - Octave up (+12 semitones, ratio = 2.0)
      - Octave down (-12 semitones, ratio = 0.5)
      - Perfect fifth (+7 semitones, ratio = 1.498)
      - Major third (+4 semitones, ratio = 1.260)
      - Sub-harmonic (-12 semitones, ratio = 0.5)
    Then blends them with the original at controlled mix levels.

Usage:
    python pipelines/eq_hypothesis/run_harmonizer_hypothesis.py \\
        --bids-root "F:\\neurophile_data\\ds003516" \\
        --subject "001" \\
        --no-ica
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
from pipelines.eq_hypothesis.run_eq_hypothesis import (
    extract_envelope,
    compute_trf_pearson,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("harmonizer_hypothesis")
mne.set_log_level("WARNING")


# ── Harmonizer Configurations to Test ────────────────────────────────────────
# Each entry: (label, list of (semitone_shift, mix_level), description)
HARMONIZER_CONFIGS = [
    (
        "Dry (No Harmonics)",
        [],
        "Original audio. No harmonics added. Baseline.",
    ),
    (
        "Octave Up",
        [(+12, 0.5)],
        "Original + 1 octave up. Hypothesis: adds 2nd harmonic (2×f₀).",
    ),
    (
        "Octave Down",
        [(-12, 0.5)],
        "Original + 1 octave down. Adds sub-harmonic. Deepens the sound.",
    ),
    (
        "Perfect Fifth",
        [(+7, 0.4)],
        "Original + perfect fifth. Most consonant interval. 3×f₀ partial.",
    ),
    (
        "Major Third",
        [(+4, 0.4)],
        "Original + major third. 5×f₀ partial. Perceived as bright and open.",
    ),
    (
        "Full Chord\n(Oct↑ + 5th + 3rd)",
        [(+12, 0.4), (+7, 0.3), (+4, 0.25)],
        "Original + octave + fifth + third. Maximally harmonic-rich stimulus.",
    ),
    (
        "Sub-Octave\n+ Oct Up",
        [(-12, 0.4), (+12, 0.4)],
        "Original + sub-octave + octave. Symmetric harmonic envelope.",
    ),
]


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """
    Pitch-shift audio by resampling (time-domain pitch shifting).
    
    Positive semitones = pitch up (faster playback, higher pitch)
    Negative semitones = pitch down (slower playback, lower pitch)
    
    Note: This is a simple resample-based method. It changes duration slightly,
    so we trim or pad to the original length.
    """
    ratio = 2.0 ** (semitones / 12.0)
    new_len = int(len(audio) / ratio)
    shifted = signal.resample(audio, new_len)
    
    # Trim or pad to original length
    if len(shifted) > len(audio):
        return shifted[: len(audio)]
    else:
        return np.pad(shifted, (0, len(audio) - len(shifted)))


def apply_harmonizer(
    audio: np.ndarray,
    harmonic_voices: list[tuple[float, float]],
) -> np.ndarray:
    """
    Apply a digital harmonizer to the audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal (float32).
    harmonic_voices : list of (semitone_shift, mix_level)
        Each entry defines one synthetic voice:
          - semitone_shift: how many semitones to shift (±12 = ±1 octave)
          - mix_level: amplitude blend ratio (0.0 = silent, 1.0 = same as dry)

    Returns
    -------
    np.ndarray
        Mixed audio (dry + all harmonic voices, normalized to original RMS).
    """
    # Start with the dry signal at unity gain
    mixed = audio.copy().astype("float64")
    
    for semitones, level in harmonic_voices:
        shifted = pitch_shift(audio, semitones).astype("float64")
        mixed += shifted * level
    
    # Normalize to original RMS so amplitude doesn't dominate the correlation
    original_rms = np.sqrt(np.mean(audio.astype("float64") ** 2))
    mixed_rms = np.sqrt(np.mean(mixed ** 2))
    if mixed_rms > 1e-10:
        mixed = mixed * (original_rms / mixed_rms)
    
    return mixed.astype("float32")


def main():
    parser = argparse.ArgumentParser(
        description="Test harmonizer hypothesis: do overtones increase EEG-audio correlation?"
    )
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--subject", type=str, default="001")
    parser.add_argument(
        "--no-ica", action="store_true",
        help="Skip MNE ICA preprocessing (faster)"
    )
    parser.add_argument(
        "--audio-file", type=Path, default=None,
        help="Override stimulus audio file (default: first sig*.mat in stimuli/)"
    )
    args = parser.parse_args()

    # ── 1. Load and preprocess EEG ────────────────────────────────────────────
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
        logger.info("Applying MNE ICA preprocessing pipeline...")
        raw = clean_raw_eeg(raw, bandpass_l=1.0, bandpass_h=40.0, n_ica_components=20)

    eeg_data = raw.get_data().T.astype("float32")  # (Time, Channels)

    # ── 2. Load audio ─────────────────────────────────────────────────────────
    audio_file = args.audio_file
    if audio_file is None:
        stim_dir = args.bids_root / "stimuli"
        audio_files = sorted(stim_dir.glob("sig*.mat"))
        if not audio_files:
            audio_files = sorted(stim_dir.glob("*.mat"))
        audio_file = audio_files[0] if audio_files else None

    if audio_file is None or not audio_file.exists():
        logger.error("No audio .mat file found. Use --audio-file to specify one.")
        return

    logger.info("Loading audio: %s", audio_file.name)
    mat = sio.loadmat(str(audio_file))
    audio_key = [k for k in mat if not k.startswith("__")][0]
    dry_audio = mat[audio_key].astype("float32").flatten()
    audio_fs = 44100.0  # Standard assumption

    # ── 3. Run each harmonizer configuration ──────────────────────────────────
    results = []
    envelopes = {}

    for label, voices, description in HARMONIZER_CONFIGS:
        clean_label = label.replace("\n", " ")
        logger.info("Processing: %s", clean_label)

        # Apply harmonizer (empty voices = dry pass-through)
        harmonized = apply_harmonizer(dry_audio, voices)

        # Extract envelope and downsample to EEG rate
        envelope = extract_envelope(harmonized, audio_fs, target_fs=eeg_fs)
        envelopes[clean_label] = envelope

        # Compute TRF correlation
        trf = compute_trf_pearson(eeg_data, envelope, lags_ms=(-200, 800), fs=eeg_fs)
        results.append({
            "label": clean_label,
            "description": description,
            "voices": voices,
            "n_harmonics": len(voices),
            **trf,
        })
        logger.info(
            "  → |Peak r|=%.4f at %.0f ms | Mean r=%.4f",
            abs(trf["peak_r"]), trf["peak_lag_ms"], trf["mean_r"],
        )

    # ── 4. Rank and print results ─────────────────────────────────────────────
    results.sort(key=lambda x: abs(x["peak_r"]), reverse=True)

    print("\n" + "=" * 72)
    print(f"HARMONIZER HYPOTHESIS RESULTS — Subject {args.subject}")
    print("=" * 72)
    print(f"{'Rank':<5} {'Configuration':<30} {'|Peak r|':>10} {'Peak Lag':>10} {'Mean r':>8}")
    print("-" * 72)
    baseline_r = abs(next(r["peak_r"] for r in results if "Dry" in r["label"]))
    for i, r in enumerate(results):
        delta = abs(r["peak_r"]) - baseline_r
        delta_str = f"(+{delta:.4f})" if delta > 0 else f"({delta:.4f})"
        winner = " ← MAX" if i == 0 else ""
        print(
            f"{i+1:<5} {r['label']:<30} {abs(r['peak_r']):>10.4f} "
            f"{r['peak_lag_ms']:>9.0f}ms {r['mean_r']:>8.4f} "
            f"{delta_str}{winner}"
        )
    print("=" * 72)

    # Print verdict
    winner = results[0]
    print(f"\n📊 VERDICT:")
    print(f"   Best config: '{winner['label']}'")
    print(f"   |Peak r| = {abs(winner['peak_r']):.4f} vs Dry baseline = {baseline_r:.4f}")
    if abs(winner["peak_r"]) > baseline_r and "Dry" not in winner["label"]:
        improvement = ((abs(winner["peak_r"]) / baseline_r) - 1) * 100
        print(f"   ✅ Hypothesis SUPPORTED: Harmonics improved correlation by {improvement:.1f}%!")
    elif "Dry" in winner["label"]:
        print(f"   ❌ Hypothesis NOT supported: Dry audio outperformed all harmonic configs.")
        print(f"   This may indicate the auditory cortex tracks the fundamental envelope rather than overtones.")
    else:
        print(f"   ⚠️  Marginal: Check mean_r for a more stable metric.")

    # ── 5. Generate plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Harmonizer Hypothesis — Does Adding Overtones Increase EEG Correlation?\n"
        f"Subject {args.subject} | Audio: {audio_file.name}",
        fontsize=13, fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot A: TRF curves per harmonizer config
    ax_trf = axes[0, 0]
    for res, color in zip(results, colors):
        lw = 2.5 if "Dry" in res["label"] else 1.0
        ls = "--" if "Dry" in res["label"] else "-"
        ax_trf.plot(res["lags_ms"], res["mean_trf"],
                    label=res["label"], color=color, linewidth=lw, linestyle=ls)
    ax_trf.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_trf.axvline(0, color="black", linewidth=0.5, linestyle=":")
    for lat, lbl, col in [(100, "N100", "red"), (200, "P200", "blue"), (300, "P300", "green")]:
        ax_trf.axvline(lat, color=col, linestyle="--", alpha=0.4)
        ax_trf.text(lat + 5, ax_trf.get_ylim()[0] * 0.8 if ax_trf.get_ylim()[0] < 0 else 0.001,
                    lbl, color=col, fontsize=8)
    ax_trf.set_xlabel("Time lag (ms)")
    ax_trf.set_ylabel("Mean Pearson r (across channels)")
    ax_trf.set_title("TRF by Harmonizer Config\n(Dry = dashed)")
    ax_trf.legend(fontsize=7, loc="upper left")

    # Plot B: |Peak r| bar chart
    ax_bar = axes[0, 1]
    labels = [r["label"] for r in results]
    peak_rs = [abs(r["peak_r"]) for r in results]
    bar_colors = ["gold" if "Dry" not in r["label"] and i == 0 else
                  ("lightcoral" if "Dry" in r["label"] else "steelblue")
                  for i, r in enumerate(results)]
    bars = ax_bar.bar(range(len(labels)), peak_rs, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax_bar.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax_bar.axhline(baseline_r, color="red", linestyle="--", linewidth=1.2, label=f"Dry baseline = {baseline_r:.4f}")
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_bar.set_ylabel("|Peak Pearson r|")
    ax_bar.set_title("Hypothesis Test: Does Harmonizer Increase Correlation?")
    ax_bar.legend(fontsize=9)

    # Plot C: Spectral content comparison (FFT of dry vs best harmonic)
    ax_fft = axes[1, 0]
    best_harmonic = next((r for r in results if "Dry" not in r["label"]), None)
    plot_dur = min(2.0, len(dry_audio) / audio_fs)
    n_fft = int(plot_dur * audio_fs)
    
    freqs = np.fft.rfftfreq(n_fft, 1.0 / audio_fs)
    dry_fft = np.abs(np.fft.rfft(dry_audio[:n_fft]))
    ax_fft.semilogy(freqs, dry_fft, color="gray", alpha=0.7, linewidth=0.8, label="Dry audio")
    
    if best_harmonic:
        best_audio = apply_harmonizer(dry_audio, best_harmonic["voices"])
        best_fft = np.abs(np.fft.rfft(best_audio[:n_fft]))
        ax_fft.semilogy(freqs, best_fft, color="gold", alpha=0.9, linewidth=0.8,
                        label=f"Best: {best_harmonic['label']}")
    
    ax_fft.set_xlim([20, 8000])
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude (log scale)")
    ax_fft.set_title("Spectral Content: Dry vs Best Harmonizer Config\n(Added overtones visible as new peaks)")
    ax_fft.legend(fontsize=9)
    # Mark harmonic fundamental multiples
    for mult, lbl in [(1, "f₀"), (2, "2f₀"), (3, "3f₀"), (4, "4f₀")]:
        ax_fft.axvline(440 * mult, color="purple", linestyle=":", alpha=0.4, linewidth=0.8)
        ax_fft.text(440 * mult + 20, dry_fft.max() * 0.5, lbl, color="purple", fontsize=7)

    # Plot D: Correlation improvement heatmap
    ax_heat = axes[1, 1]
    trf_matrix = np.array([r["mean_trf"] for r in results])
    lag_axis = results[0]["lags_ms"]
    im = ax_heat.imshow(
        np.abs(trf_matrix), aspect="auto", origin="upper", cmap="YlOrRd",
        extent=[lag_axis[0], lag_axis[-1], len(results) - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax_heat, label="|Pearson r|")
    ax_heat.set_yticks(range(len(results)))
    ax_heat.set_yticklabels([r["label"] for r in results], fontsize=8)
    ax_heat.set_xlabel("Time lag (ms)")
    ax_heat.set_title("TRF Heatmap: |r| across Lag × Harmonizer Config")
    for lat, lbl, col in [(100, "N100", "white"), (200, "P200", "cyan"), (300, "P300", "lime")]:
        ax_heat.axvline(lat, color=col, linestyle="--", alpha=0.7, linewidth=1.0)
        ax_heat.text(lat + 5, 0.2, lbl, color=col, fontsize=7)

    plt.tight_layout()

    out_path = Path("checkpoints") / f"harmonizer_hypothesis_sub{args.subject}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", out_path)

    print(f"\nFigure saved → {out_path}")
    print("Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
