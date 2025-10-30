"""
ITERATIVE TRAINING WITH DATA REGENERATION
==========================================

Trains model iteratively until good accuracy is achieved on all classes.

Strategy:
- Train for 100 epochs
- Evaluate on test set
- If accuracy < target: regenerate data and train again
- Repeat up to 100 iterations
- Save best model

Usage:
    python iterative_training.py --max_iterations 100 --target_accuracy 0.95
"""

import argparse
import numpy as np
from pathlib import Path

from train_spt_classifier import SPTClassifierTrainer


def iterative_training(
    max_iterations: int = 100,
    target_overall_accuracy: float = 0.95,
    target_per_class_accuracy: float = 0.90,
    n_samples_per_class: int = 2000,
    epochs_per_iteration: int = 100,
    mode: str = 'both',
    ratio_3d: float = 0.5,
    output_dir: str = './spt_iterative_model',
    verbose: bool = True
):
    """
    Iterative training with automatic data regeneration

    Args:
        max_iterations: Maximum number of training iterations
        target_overall_accuracy: Target overall accuracy (e.g., 0.95 = 95%)
        target_per_class_accuracy: Minimum accuracy for each class
        n_samples_per_class: Samples per class per iteration (keep small for speed!)
        epochs_per_iteration: Epochs per iteration (e.g., 100)
        mode: '2D', '3D', or 'both'
        ratio_3d: Ratio of 3D data if mode='both'
        output_dir: Where to save best model
        verbose: Print progress
    """

    if verbose:
        print("="*80)
        print("ITERATIVE TRAINING WITH DATA REGENERATION")
        print("="*80)
        print(f"Max Iterations: {max_iterations}")
        print(f"Epochs per Iteration: {epochs_per_iteration}")
        print(f"Samples per Class: {n_samples_per_class}")
        print(f"Target Overall Accuracy: {target_overall_accuracy:.1%}")
        print(f"Target Per-Class Accuracy: {target_per_class_accuracy:.1%}")
        print(f"Mode: {mode} (3D ratio: {ratio_3d:.0%})")
        print("="*80 + "\n")

    best_overall_accuracy = 0.0
    best_model = None
    best_iteration = 0

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print("\n" + "="*80)
            print(f"ITERATION {iteration}/{max_iterations}")
            print("="*80 + "\n")

        # Create trainer
        trainer = SPTClassifierTrainer(
            max_length=500,
            output_dir=output_dir,
            random_seed=42 + iteration  # Different seed each iteration!
        )

        # Generate NEW data each iteration
        if verbose:
            print(f"[Iteration {iteration}] Generating fresh training data...")

        trainer.generate_training_data(
            n_samples_per_class=n_samples_per_class,
            mode=mode,
            ratio_3d=ratio_3d,
            polymerization_degree=0.5,  # Will be overridden by augmentation
            verbose=verbose
        )

        # Build model
        if verbose:
            print(f"\n[Iteration {iteration}] Building model...")
        trainer.build_model(verbose=False)

        # Train
        if verbose:
            print(f"\n[Iteration {iteration}] Training for {epochs_per_iteration} epochs...")

        trainer.train(
            epochs=epochs_per_iteration,
            batch_size=512,
            verbose=1
        )

        # Evaluate
        if verbose:
            print(f"\n[Iteration {iteration}] Evaluating...")

        overall_acc, report, cm = trainer.evaluate(verbose=True)

        # Calculate per-class accuracy from confusion matrix
        per_class_acc = {}
        for i, class_name in enumerate(trainer.class_names):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            per_class_acc[class_name] = class_correct / class_total if class_total > 0 else 0.0

        min_class_acc = min(per_class_acc.values())

        if verbose:
            print(f"\n[Iteration {iteration}] RESULTS:")
            print(f"  Overall Accuracy: {overall_acc:.1%}")
            print(f"  Per-Class Accuracies:")
            for class_name, acc in per_class_acc.items():
                status = "OK" if acc >= target_per_class_accuracy else "LOW"
                print(f"    {class_name}: {acc:.1%} [{status}]")
            print(f"  Minimum Class Accuracy: {min_class_acc:.1%}")

        # Track best
        if overall_acc > best_overall_accuracy:
            best_overall_accuracy = overall_acc
            best_model = trainer
            best_iteration = iteration

            if verbose:
                print(f"\n  NEW BEST MODEL! Accuracy: {overall_acc:.1%}")

        # Check if target reached
        if overall_acc >= target_overall_accuracy and min_class_acc >= target_per_class_accuracy:
            if verbose:
                print("\n" + "="*80)
                print("TARGET ACCURACY REACHED!")
                print("="*80)
                print(f"Iteration: {iteration}/{max_iterations}")
                print(f"Overall Accuracy: {overall_acc:.1%} >= {target_overall_accuracy:.1%}")
                print(f"Min Class Accuracy: {min_class_acc:.1%} >= {target_per_class_accuracy:.1%}")
                print("="*80 + "\n")

            # Save best model
            if verbose:
                print("Saving model...")
            trainer.save_model(verbose=True)

            return trainer, None, iteration

        # Continue to next iteration with fresh data
        if verbose:
            print(f"\n[Iteration {iteration}] Target not reached. Generating new data for next iteration...")

    # Max iterations reached - save best model
    if verbose:
        print("\n" + "="*80)
        print("MAX ITERATIONS REACHED")
        print("="*80)
        print(f"Best Iteration: {best_iteration}/{max_iterations}")
        print(f"Best Overall Accuracy: {best_overall_accuracy:.1%}")
        print("="*80 + "\n")

    if best_model is not None:
        if verbose:
            print("Saving best model...")
        best_model.save_model(verbose=True)
        return best_model, None, best_iteration
    else:
        print("WARNING: No model was trained successfully!")
        return None, None, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iterative training with data regeneration')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--target_accuracy', type=float, default=0.95,
                        help='Target overall accuracy (default: 0.95)')
    parser.add_argument('--target_per_class', type=float, default=0.90,
                        help='Minimum per-class accuracy (default: 0.90)')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Samples per class (default: 2000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs per iteration (default: 100)')
    parser.add_argument('--mode', type=str, default='both', choices=['2D', '3D', 'both'],
                        help='Dimensionality mode (default: both)')
    parser.add_argument('--output', type=str, default='./spt_iterative_model',
                        help='Output directory (default: ./spt_iterative_model)')

    args = parser.parse_args()

    trainer, metrics, final_iteration = iterative_training(
        max_iterations=args.max_iterations,
        target_overall_accuracy=args.target_accuracy,
        target_per_class_accuracy=args.target_per_class,
        n_samples_per_class=args.samples,
        epochs_per_iteration=args.epochs,
        mode=args.mode,
        ratio_3d=0.5,
        output_dir=args.output,
        verbose=True
    )

    if trainer is not None:
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Final Iteration: {final_iteration}")
        print(f"Model saved to: {args.output}/")
        print("="*80)
    else:
        print("\nERROR: Training failed!")
