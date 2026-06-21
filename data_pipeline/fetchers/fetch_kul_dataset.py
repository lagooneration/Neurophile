"""
fetch_kul_dataset.py
=====================
Download the KU Leuven multi-speaker auditory attention EEG dataset
from Zenodo using `pooch` (checksum-verified download).

Dataset
-------
Biesmans et al. (2017) / Vandecappelle et al. (2021)
"KU Leuven EEG Dataset for Auditory Attention Detection"

Zenodo DOI: 10.5281/zenodo.1199011
  - EEG recorded from 16 subjects, two competing speakers, 64 channels
  - Stimuli: Flemish speech, 50 Hz sampling rate (EEG)

Usage
-----
    python data_pipeline/fetchers/fetch_kul_dataset.py
    python data_pipeline/fetchers/fetch_kul_dataset.py --output-dir ./data/raw/kul
    python data_pipeline/fetchers/fetch_kul_dataset.py --dry-run  # check URLs only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Zenodo record registry ─────────────────────────────────────────────────────
# Files from Zenodo record 1199011 (KU Leuven AAD dataset)
# SHA-256 hashes obtained from the Zenodo metadata API.
# Update these if Zenodo releases a new dataset version.
_ZENODO_BASE = "https://zenodo.org/record/1199011/files"

_KUL_FILES: list[dict] = [
    {
        "filename": "DATA_preproc.zip",
        "url": f"{_ZENODO_BASE}/DATA_preproc.zip",
        "sha256": "md5:1c1ca1390ad99ff41c33f36ca67f8418",
        "size_mb": 1753.8,
    }
]

_DEFAULT_OUTPUT = Path("./data/raw/kul")


def _check_pooch() -> None:
    try:
        import pooch  # noqa: F401
    except ImportError:
        logger.error(
            "pooch is required: pip install pooch\n"
            "(Already listed in neurophile core dependencies.)"
        )
        sys.exit(1)


def fetch_kul_dataset(
    output_dir: Path = _DEFAULT_OUTPUT,
    dry_run: bool = False,
    subjects: list[int] | None = None,
) -> list[Path]:
    """Download the KU Leuven AAD dataset to ``output_dir``.

    Parameters
    ----------
    output_dir : Path
        Destination directory (created if absent).
    dry_run : bool
        If True, print URLs and sizes without downloading.
    subjects : list[int] or None
        Note: The entire dataset is bundled in a single 1.7GB ZIP file on Zenodo.
        This argument is currently ignored; the whole bundle is downloaded.

    Returns
    -------
    downloaded : list[Path]
        Paths of downloaded files.
    """
    _check_pooch()
    import pooch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        total_mb = sum(f["size_mb"] for f in _KUL_FILES)
        logger.info("DRY RUN — 1 bundle, ~%.1f MB total:", total_mb)
        for f in _KUL_FILES:
            logger.info("  %s  (%s)", f["url"], f["filename"])
        return []

    downloaded: list[Path] = []
    registry = pooch.create(
        path=output_dir,
        base_url=_ZENODO_BASE + "/",
        registry={f["filename"]: f["sha256"] for f in _KUL_FILES},
    )

    for file_meta in _KUL_FILES:
        fname = file_meta["filename"]
        logger.info("Fetching %s (~%.1f MB). This might take a while...", fname, file_meta["size_mb"])
        try:
            # Download and automatically extract the ZIP file
            local_paths = registry.fetch(fname, processor=pooch.Unzip())
            
            if isinstance(local_paths, list):
                downloaded.extend([Path(p) for p in local_paths])
            else:
                downloaded.append(Path(local_paths))
                
            logger.info("  ✓ Successfully downloaded and extracted %s", fname)
        except Exception as exc:
            logger.error("  ✗ Failed to download %s: %s", fname, exc)

    logger.info(
        "KUL dataset: extracted %d files to %s",
        len(downloaded), output_dir,
    )
    return downloaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the KU Leuven AAD EEG dataset from Zenodo."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Destination directory (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", metavar="N",
        help="Subject IDs to download (1-indexed). Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print URLs and estimated sizes without downloading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_kul_dataset(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        subjects=args.subjects,
    )
