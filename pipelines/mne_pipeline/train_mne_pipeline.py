"""
pipelines/mne_pipeline/train_mne_pipeline.py

A full end-to-end training script using the pure Python MNE pipeline.
This is the MATLAB/EEGLAB-free version of scripts/train_bids_real.py.

Pipeline stages per subject:
  1. Load raw .set from BIDS dataset (mne.io.read_raw_eeglab)
  2. Preprocess with MNE (bandpass, bad channel interpolation, ICA via mne-icalabel)
  3. Convert to numpy trials
  4. Update Global CI model weights on GPU

Usage:
    python pipelines/mne_pipeline/train_mne_pipeline.py \\
        --bids-root "F:\\neurophile_data\\ds003516" \\
        --subject "all" \\
        --device cuda \\
        --epochs 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mne
import numpy as np

# Allow importing from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from neurophile.models import MesgaraniAdapter, GlobalCITrainer
from pipelines.mne_pipeline.preprocess import clean_raw_eeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("mne_pipeline.train")


def load_and_preprocess(
    bids_root: Path,
    subject: str,
    enable_preprocessing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a subject's EEG from the BIDS dataset, optionally preprocess with
    the full MNE ICA pipeline, and convert into windowed numpy trials.

    Returns (eeg_trials, env_trials, labels).
    """
    import scipy.io as sio

    # Locate the .set file directly using pathlib (Windows-safe)
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    if not set_files:
        raise FileNotFoundError(f"No .set file found in {eeg_dir}")

    logger.info("Loading: %s", set_files[0])
    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    if enable_preprocessing:
        raw = clean_raw_eeg(raw, bandpass_l=1.0, bandpass_h=40.0, n_ica_components=20)
    else:
        logger.info("Preprocessing skipped (--no-preprocess flag set).")

    # ── Extract EEG data ───────────────────────────────────────────────────────
    eeg_data = raw.pick_types(eeg=True).get_data().T.astype("float32")  # (T, C)

    # Enforce 64 channels
    if eeg_data.shape[1] > 64:
        eeg_data = eeg_data[:, :64]
    elif eeg_data.shape[1] < 64:
        eeg_data = np.pad(eeg_data, ((0, 0), (0, 64 - eeg_data.shape[1])))

    # NaN guard (safety net)
    np.nan_to_num(eeg_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Load stimulus envelope ─────────────────────────────────────────────────
    stim_files = list((bids_root / "stimuli").glob("*.mat"))
    if stim_files:
        mat = sio.loadmat(stim_files[0])
        key = [k for k in mat if not k.startswith("__")][0]
        stim = mat[key].astype("float32").flatten()
        if stim.shape[0] < eeg_data.shape[0]:
            stim = np.pad(stim, (0, eeg_data.shape[0] - stim.shape[0]))
        env_data = stim[: eeg_data.shape[0]].reshape(-1, 1)
    else:
        logger.warning("No stimuli found — using dummy envelope.")
        env_data = np.random.randn(eeg_data.shape[0], 1).astype("float32")

    # ── Window into 512-sample trials ─────────────────────────────────────────
    window_t = 512
    n_trials = eeg_data.shape[0] // window_t
    eeg_trials = np.stack([eeg_data[i * window_t: (i + 1) * window_t] for i in range(n_trials)])
    env_trials = np.stack([env_data[i * window_t: (i + 1) * window_t] for i in range(n_trials)])
    labels = np.array([float(i % 2) for i in range(n_trials)], dtype="float32")

    return eeg_trials, env_trials, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNE-based global CI model trainer (no MATLAB required)"
    )
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument(
        "--subject",
        type=str,
        default="001",
        help="Subject ID or 'all' to iterate over all 25 subjects",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip MNE preprocessing (useful for already-cleaned data)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = (
        [f"{i:03d}" for i in range(1, 26)]
        if args.subject.lower() == "all"
        else [args.subject]
    )

    logger.info(
        "MNE Pipeline | %d subject(s) | device=%s | epochs=%d",
        len(subjects), args.device, args.epochs,
    )

    # Initialise the global model ONCE — weights accumulate across subjects
    model = MesgaraniAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    trainer = GlobalCITrainer(
        model=model,
        epochs=args.epochs,
        batch_size=8,
        device=args.device,
        output_dir=args.output_dir,
    )

    t_global = time.time()

    for sub in subjects:
        logger.info("=" * 50)
        logger.info("Subject %s", sub)

        try:
            eeg, env, labels = load_and_preprocess(
                args.bids_root, sub,
                enable_preprocessing=not args.no_preprocess,
            )
        except Exception as exc:
            logger.error("Skipping subject %s: %s", sub, exc)
            continue

        n = len(eeg)
        split = max(1, int(n * (1 - args.test_split)))
        logger.info("Train=%d | Test=%d", split, n - split)

        t0 = time.time()
        trainer.train(eeg[:split], env[:split], labels[:split])
        logger.info("Subject %s done in %.1fs", sub, time.time() - t0)

    logger.info("=" * 50)
    logger.info(
        "All subjects complete in %.1fs. Checkpoint saved to %s",
        time.time() - t_global,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
