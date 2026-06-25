"""
scripts/eeglab_bridge.py

A Python wrapper that triggers the MATLAB EEGLAB backend silently.
This script uses Python's subprocess module to execute the `run_eeglab_ica.m`
function without ever opening the MATLAB GUI, effectively chaining the EEGLAB
pipeline into the PyTorch workflow.
"""

import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("eeglab_bridge")

def run_eeglab_via_cli(input_file: Path, output_file: Path) -> bool:
    """Fires a background MATLAB process to run EEGLAB's ICA."""
    
    # We must ensure we are running the script from the directory where the .m file lives
    script_dir = Path(__file__).resolve().parent
    
    # Build the MATLAB CLI command
    # -batch executes the string and then silently exits
    matlab_command = f"run_eeglab_ica('{input_file}', '{output_file}')"
    
    cli_args = [
        "matlab", 
        "-batch", 
        matlab_command
    ]
    
    logger.info("Triggering MATLAB EEGLAB backend...")
    logger.info("Executing: %s", " ".join(cli_args))
    
    try:
        # Run the subprocess in the script's directory so MATLAB finds run_eeglab_ica.m
        result = subprocess.run(cli_args, cwd=script_dir, text=True, capture_output=True)
        
        # Print MATLAB's console output so the Python user can see what EEGLAB is doing
        if result.stdout:
            print(result.stdout)
            
        if result.returncode == 0:
            logger.info("EEGLAB ICA processing completed successfully!")
            return True
        else:
            logger.error("EEGLAB failed with exit code %d", result.returncode)
            if result.stderr:
                logger.error("MATLAB Stderr: %s", result.stderr)
            return False
            
    except FileNotFoundError:
        logger.error("Could not find 'matlab' command. Ensure MATLAB is installed and in your system PATH.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Python-to-MATLAB EEGLAB CLI Bridge")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw .set file")
    parser.add_argument("--output", type=Path, required=True, help="Path to save cleaned .set file")
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error("Input file %s does not exist!", args.input)
        return
        
    run_eeglab_via_cli(args.input, args.output)

if __name__ == "__main__":
    main()
