"""Utility script to run inference with a trained SPT classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from train_spt_classifier import SPTClassifierTrainer


def load_trajectories(npz_path: Path) -> List[np.ndarray]:
    """Load trajectories from an ``.npz`` file.

    The file must contain an array named ``trajectories`` of shape
    ``(n_samples, n_steps, dim)``.  Any trailing zero padding is accepted.
    """

    with np.load(npz_path, allow_pickle=True) as data:
        if "trajectories" not in data:
            raise KeyError("NPZ file must contain an array named 'trajectories'.")
        arr = data["trajectories"]
    trajectories: List[np.ndarray] = []
    for item in arr:
        item = np.asarray(item)
        # Remove trailing zero padding if present
        if item.ndim == 2:
            mask = np.any(item != 0.0, axis=1)
            if mask.any():
                last_index = np.max(np.where(mask)) + 1
                item = item[:last_index]
        trajectories.append(item)
    return trajectories


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run inference using a trained SPT classifier")
    parser.add_argument("--artifacts", required=True, type=Path, help="Directory containing saved model artifacts")
    parser.add_argument(
        "--npz",
        type=Path,
        help="Optional NPZ file with a 'trajectories' array. If omitted a synthetic demo batch is generated.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON file to store predictions")
    args = parser.parse_args()

    trainer = SPTClassifierTrainer.from_artifacts(args.artifacts)

    if args.npz:
        trajectories = load_trajectories(args.npz)
    else:
        from spt_trajectory_generator import generate_spt_dataset

        demo_traj, _, _, _ = generate_spt_dataset(
            n_samples_per_class=4,
            min_length=80,
            max_length=trainer.max_length,
            dimensionality="2D",
            polymerization_degree=0.5,
            verbose=False,
        )
        trajectories = demo_traj

    labels, probs = trainer.predict_trajectories(trajectories)

    for idx, (label, prob) in enumerate(zip(labels, probs)):
        best = float(np.max(prob))
        print(f"Sample {idx:02d}: {label} (confidence {best:.3f})")

    if args.output:
        payload = {"labels": labels, "probabilities": probs.tolist(), "class_names": trainer.class_names}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    run_cli()
