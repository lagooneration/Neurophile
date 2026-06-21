"""
fetch_openneuro_bids.py
========================
Download BIDS-formatted AAD EEG datasets from OpenNeuro.

OpenNeuro hosts several publicly available EEG datasets relevant to
auditory attention decoding. This script supports `aws s3` (fastest)
and HTTPS fallback via `openneuro-py`.

Supported Datasets
------------------
ds003516 : "Neural Dynamics of Attention in Natural Listening"
           Zion-Golumbic lab — EEG + eye-tracking during cocktail party
ds002034 : "BIDS EEG for Auditory Attention Decoding"
           Useful baseline for multi-speaker attention studies

Usage
-----
    python data_pipeline/fetchers/fetch_openneuro_bids.py --dataset ds003516
    python data_pipeline/fetchers/fetch_openneuro_bids.py --dataset ds003516 \\
        --subject sub-01 --output-dir ./data/raw/openneuro
    python data_pipeline/fetchers/fetch_openneuro_bids.py --list-datasets
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Known datasets registry ────────────────────────────────────────────────────
_OPENNEURO_BASE = "s3://openneuro.org"

_DATASETS: dict[str, dict] = {
    "ds003516": {
        "name": "Neural Dynamics of Selective Attention (Zion-Golumbic)",
        "s3_path": f"{_OPENNEURO_BASE}/ds003516",
        "doi": "10.18112/openneuro.ds003516.v1.0.0",
        "n_subjects": 20,
        "description": (
            "EEG recorded during natural listening to competing speakers. "
            "Includes cross-attention paradigm relevant to Zion-Golumbic adapter."
        ),
    },
    "ds002034": {
        "name": "AAD Baseline EEG (multi-speaker)",
        "s3_path": f"{_OPENNEURO_BASE}/ds002034",
        "doi": "10.18112/openneuro.ds002034.v1.0.1",
        "n_subjects": 18,
        "description": (
            "Dichotic listening EEG, 64-ch, standard cocktail-party paradigm."
        ),
    },
}

_DEFAULT_OUTPUT = Path("./data/raw/openneuro")


def list_datasets() -> None:
    """Print available datasets to stdout."""
    print("\nAvailable OpenNeuro AAD datasets:")
    print("-" * 60)
    for ds_id, meta in _DATASETS.items():
        print(f"  {ds_id}  —  {meta['name']}")
        print(f"           {meta['description']}")
        print(f"           DOI: {meta['doi']}")
    print()


def fetch_openneuro_dataset(
    dataset_id: str,
    output_dir: Path = _DEFAULT_OUTPUT,
    subject: str | None = None,
    dry_run: bool = False,
    use_aws: bool = True,
) -> bool:
    """Download an OpenNeuro BIDS dataset.

    Parameters
    ----------
    dataset_id : str
        OpenNeuro dataset identifier (e.g. ``"ds003516"``).
    output_dir : Path
        Root directory for downloaded data.
    subject : str or None
        Specific subject folder to download (e.g. ``"sub-01"``).
        None = download entire dataset.
    dry_run : bool
        Print the aws command without executing.
    use_aws : bool
        Use AWS CLI (fast, parallel). If False, falls back to openneuro-py.

    Returns
    -------
    success : bool
    """
    if dataset_id not in _DATASETS:
        logger.error(
            "Unknown dataset: %s. Available: %s",
            dataset_id, list(_DATASETS.keys()),
        )
        return False

    meta = _DATASETS[dataset_id]
    dest = Path(output_dir) / dataset_id
    dest.mkdir(parents=True, exist_ok=True)

    s3_src = meta["s3_path"]
    aws_dest = dest
    if subject:
        s3_src = f"{s3_src}/{subject}"
        aws_dest = dest / subject

    if use_aws:
        return _fetch_via_aws(s3_src, aws_dest, dry_run)
    return _fetch_via_openneuro_py(dataset_id, dest, subject, dry_run)


def _fetch_via_aws(s3_src: str, dest: Path, dry_run: bool) -> bool:
    """Download using AWS CLI (fastest method — parallel, no auth required)."""
    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",  # public bucket
        s3_src,
        str(dest),
    ]

    if dry_run:
        cmd.append("--dryrun")

    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, text=True)
        logger.info("AWS sync complete → %s", dest)
        return result.returncode == 0
    except FileNotFoundError:
        logger.warning("AWS CLI not found. Falling back to openneuro-py.")
        return _fetch_via_openneuro_py(
            s3_src.split("/")[-1], dest, subject=None, dry_run=dry_run
        )
    except subprocess.CalledProcessError as exc:
        logger.error("aws s3 sync failed: %s", exc)
        return False


def _fetch_via_openneuro_py(
    dataset_id: str,
    dest: Path,
    subject: str | None,
    dry_run: bool,
) -> bool:
    """Fallback using the openneuro-py package."""
    try:
        import openneuro  # type: ignore[import]
    except ImportError:
        logger.error(
            "Neither AWS CLI nor openneuro-py is available.\n"
            "Install one of:\n"
            "  - AWS CLI: https://aws.amazon.com/cli/\n"
            "  - openneuro-py: pip install openneuro-py"
        )
        return False

    if dry_run:
        logger.info("DRY RUN: would download %s → %s", dataset_id, dest)
        return True

    logger.info("Downloading %s via openneuro-py …", dataset_id)
    include = [subject] if subject else None
    openneuro.download(dataset=dataset_id, target_dir=str(dest), include=include)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BIDS AAD EEG datasets from OpenNeuro."
    )
    parser.add_argument("--dataset", type=str, help="OpenNeuro dataset ID (e.g. ds003516)")
    parser.add_argument("--subject", type=str, default=None, help="Specific subject (e.g. sub-01)")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-aws", action="store_true", help="Use openneuro-py instead of AWS CLI")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.list_datasets:
        list_datasets()
        sys.exit(0)
    if not args.dataset:
        print("Error: --dataset is required. Use --list-datasets to see options.")
        sys.exit(1)
    success = fetch_openneuro_dataset(
        dataset_id=args.dataset,
        output_dir=args.output_dir,
        subject=args.subject,
        dry_run=args.dry_run,
        use_aws=not args.no_aws,
    )
    sys.exit(0 if success else 1)
