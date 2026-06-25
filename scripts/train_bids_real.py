"""
scripts/train_bids_real.py

Trains the Global CI Foundation model natively on OpenNeuro BIDS datasets
(like ds003516) stored on an external drive.
Supports optional inline EEGLAB ICA artifact rejection (Option B) via
the --use-eeglab flag, which fires a silent MATLAB subprocess per subject.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from mne_bids import BIDSPath, read_raw_bids
except ImportError:
    print("mne-bids is required. Please install it via: pip install mne-bids")
    sys.exit(1)

from neurophile.models import MesgaraniAdapter, GlobalCITrainer
from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline, CIArtifactConfig
from neurophile.preprocessing.ci_artifact.ica_cancellation import ICACancellationConfig

# Import the MATLAB EEGLAB CLI bridge
sys.path.append(str(Path(__file__).resolve().parent))
from eeglab_bridge import run_eeglab_via_cli

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("train_bids_real")

def load_bids_data(
    bids_root: Path,
    subject: str,
    enable_ica: bool = False,
    use_eeglab: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads EEG and Audio from an OpenNeuro BIDS dataset and optionally cleans artifacts.
    
    If use_eeglab=True, fires a background MATLAB EEGLAB subprocess to run runica ICA
    on the raw .set file and saves a cleaned copy before MNE reads it.
    """
    logger.info("Scanning BIDS root: %s for subject %s", bids_root, subject)
    
    # Locate the raw .set file using standard pathlib (avoids Windows backslash bug in match())
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    if not set_files:
        raise ValueError(f"No .set EEG data found for subject {subject} in {eeg_dir}")

    raw_set_file = set_files[0]

    # --- OPTION B: INLINE EEGLAB ICA VIA MATLAB SUBPROCESS ---
    if use_eeglab:
        cleaned_set_file = eeg_dir / f"sub-{subject}_eeg_cleaned.set"
        if cleaned_set_file.exists():
            logger.info("Found cached EEGLAB-cleaned file: %s", cleaned_set_file)
        else:
            logger.info("Running EEGLAB ICA silently via MATLAB for subject %s...", subject)
            t_eeg = time.time()
            success = run_eeglab_via_cli(raw_set_file, cleaned_set_file)
            if success:
                logger.info("EEGLAB ICA complete in %.1f seconds!", time.time() - t_eeg)
            else:
                logger.warning("EEGLAB ICA failed! Falling back to raw data for subject %s.", subject)
                cleaned_set_file = raw_set_file
        # Load the EEGLAB-cleaned file directly via MNE
        import mne
        raw = mne.io.read_raw_eeglab(str(cleaned_set_file), preload=True, verbose=False)
    else:
        # Standard MNE-BIDS path for non-EEGLAB subjects
        bids_path = BIDSPath(
            subject=subject,
            task="AttendedSpeakerParadigmOwnName",
            datatype="eeg",
            suffix="eeg",
            extension=".set",
            root=bids_root
        )
        bids_path.update(check=False)
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
    
    # Typically, EEG channels are marked as 'eeg', and stimuli as 'stim' or 'misc'
    eeg_picks = raw.copy().pick_types(eeg=True).get_data() # shape: (Channels, Time)
    
    # Let's transpose to (Time, Channels) for the model
    eeg_data = eeg_picks.T.astype("float32")
    
    # Keep up to 64 channels
    if eeg_data.shape[1] > 64:
        eeg_data = eeg_data[:, :64]
    elif eeg_data.shape[1] < 64:
        # Pad with zeros if less than 64 channels
        pad_width = 64 - eeg_data.shape[1]
        eeg_data = np.pad(eeg_data, ((0,0), (0, pad_width)))
        
    # --- SCRUB NANS (Bad Channels) ---
    nan_count = np.isnan(eeg_data).sum()
    if nan_count > 0:
        logger.warning("Found %d NaN values in EEG! Scrubbing bad channels to 0.0 before ICA...", nan_count)
        np.nan_to_num(eeg_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
    # --- PREPROCESSING (ICA) ---
    if enable_ica:
        logger.info("Applying CIArtifactPipeline (ICA) to clean EEG data...")
        config = CIArtifactConfig(
            stage3_enabled=True,
            stage3=ICACancellationConfig(kurtosis_threshold=5.0)
        )
        # CIArtifactPipeline takes (Time, Channels) array, or list of arrays.
        pipeline = CIArtifactPipeline(fs=raw.info['sfreq'], config=config)
        eeg_data = pipeline.run(eeg_data)
        logger.info("ICA Cleaning complete!")
        
    # In OpenNeuro datasets, the audio/stimulus is often embedded as a 'misc' or 'stim' channel
    # Or sometimes we just use synthetic envelopes if the stimulus isn't mapped properly yet.
    # For this baseline script, we extract whatever 'stim' or 'misc' channels exist.
    import scipy.io as sio
    # Try to load a real stimulus envelope instead of dummy noise
    stim_files = list((bids_root / "stimuli").glob("*.mat"))
    if stim_files:
        logger.info("Loading real audio envelope from %s", stim_files[0])
        mat = sio.loadmat(stim_files[0])
        # Find the first key that isn't a dunder
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        stim_data = mat[key].astype("float32")
        
        # Resample or slice to match EEG length
        if stim_data.shape[0] < eeg_data.shape[0]:
            # pad
            pad_width = eeg_data.shape[0] - stim_data.shape[0]
            env_data = np.pad(stim_data.reshape(-1, 1), ((0, pad_width), (0, 0)))
        else:
            env_data = stim_data[:eeg_data.shape[0]].reshape(-1, 1)
    else:
        logger.warning("Could not find stimuli in BIDS. Generating dummy envelope.")
        env_data = np.random.randn(eeg_data.shape[0], 1).astype("float32")
        
    # Create windowed trials (e.g. 512 steps per trial)
    window_t = 512
    n_trials = len(eeg_data) // window_t
    
    eeg_trials = []
    env_trials = []
    labels = []
    
    for i in range(n_trials):
        start = i * window_t
        end = start + window_t
        
        eeg_trials.append(eeg_data[start:end])
        env_trials.append(env_data[start:end])
        labels.append(float(i % 2)) # Dummy labels alternating 0 and 1
        
    return np.stack(eeg_trials), np.stack(env_trials), np.array(labels, dtype="float32")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True, help="Path to the downloaded OpenNeuro dataset")
    parser.add_argument("--subject", type=str, default="001", help="Subject ID or 'all' to train on entire dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--enable-ica", action="store_true", help="Enable Python CIArtifactPipeline ICA")
    parser.add_argument("--use-eeglab", action="store_true", help="Run MATLAB EEGLAB runica ICA inline before each subject (slower but more accurate)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.subject.lower() == "all":
        # ds003516 has 25 subjects
        subjects = [f"{i:03d}" for i in range(1, 26)]
        logger.info("Sequential Deep Learning mode activated. Will iterate over %d subjects.", len(subjects))
    else:
        subjects = [args.subject]
        
    # Initialize the global model ONCE
    model = MesgaraniAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    trainer = GlobalCITrainer(
        model=model,
        epochs=args.epochs,
        batch_size=8,
        device=args.device,
        output_dir=args.output_dir
    )
    
    t_global = time.time()
    
    for sub in subjects:
        logger.info("="*50)
        logger.info("Starting training for Subject %s...", sub)
        
        try:
            train_eeg, train_env, train_label = load_bids_data(
                args.bids_root, sub,
                enable_ica=args.enable_ica,
                use_eeglab=args.use_eeglab
            )
        except Exception as e:
            logger.error("Skipping subject %s due to error: %s", sub, e)
            continue
        
        n_trials = len(train_eeg)
        split_idx = int(n_trials * (1 - args.test_split))
        if split_idx == 0: split_idx = 1
        
        test_eeg, test_env, test_label = train_eeg[split_idx:], train_env[split_idx:], train_label[split_idx:]
        train_eeg, train_env, train_label = train_eeg[:split_idx], train_env[:split_idx], train_label[:split_idx]
        
        logger.info("Train samples: %d | Test samples: %d", len(train_eeg), len(test_eeg))
        
        # Train updates the same global weights
        t0 = time.time()
        trainer.train(train_eeg, train_env, train_label)
        logger.info("Subject %s complete in %.1f seconds.", sub, time.time() - t0)
        
    logger.info("="*50)
    logger.info("Global model training across all subjects completed in %.1f seconds!", time.time() - t_global)

if __name__ == "__main__":
    main()
