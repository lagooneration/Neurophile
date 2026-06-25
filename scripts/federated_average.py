"""
scripts/federated_average.py

Averages the state_dict of multiple .pt checkpoints into a single global model.
This script demonstrates a local Federated Learning aggregation step.
"""
import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def federated_average(checkpoint_paths: list[Path], output_path: Path):
    if not checkpoint_paths:
        logger.error("No checkpoints provided for federated averaging.")
        return
        
    logger.info("Starting Federated Averaging (FedAvg) on %d models...", len(checkpoint_paths))
    
    # Load all state dicts
    state_dicts = []
    for cp in checkpoint_paths:
        logger.info("Loading %s", cp.name)
        loaded = torch.load(cp, map_location="cpu", weights_only=True)
        # Extract model_state if wrapped by GlobalCITrainer
        if isinstance(loaded, dict) and "model_state" in loaded:
            state_dicts.append(loaded["model_state"])
        else:
            state_dicts.append(loaded)
        
    # Assume all checkpoints have the same architecture / keys
    keys = list(state_dicts[0].keys())
    
    # Check that architectures match
    for sd in state_dicts[1:]:
        if list(sd.keys()) != keys:
            logger.error("Architecture mismatch! You cannot average a CNN with a CRN. Checkpoints must have the exact same architecture.")
            return
    
    # Average weights
    global_dict = {}
    for key in keys:
        # Sum the weights for this key across all state_dicts
        tensor_sum = torch.stack([sd[key] for sd in state_dicts]).sum(dim=0)
        # Average
        global_dict[key] = tensor_sum / len(state_dicts)
        
    logger.info("Averaged %d parameters.", len(keys))
    
    # Save the global model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_dict, output_path)
    logger.info("Federated Global Model saved successfully to: %s", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average multiple .pt checkpoints.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, 
                        help="List of .pt files to average")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/global_federated_model.pt"),
                        help="Path to save the averaged model")
    args = parser.parse_args()
    
    federated_average(args.checkpoints, args.output)
