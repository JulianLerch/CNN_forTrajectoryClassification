"""
PREDICTION SCRIPT FOR TRAINED SPT CLASSIFIER
==============================================

This script shows how to use a trained model to predict diffusion types
for new trajectories.

Usage:
------
python predict_trajectories.py --model_dir ./spt_trained_model_app

Or import and use in your own scripts:
    from predict_trajectories import SPTPredictor
    predictor = SPTPredictor('./spt_trained_model_app')
    predictions = predictor.predict(trajectories, D_values, poly_degrees)
"""

import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import sys

from train_spt_classifier import SPTClassifierTrainer
from spt_trajectory_generator import generate_spt_dataset


class SPTPredictor:
    """
    Convenient wrapper for making predictions with trained SPT classifier
    """

    def __init__(self, model_dir: str, verbose: bool = True):
        """
        Load trained model

        Args:
            model_dir: Directory containing trained model files
            verbose: Print loading info
        """
        self.model_dir = model_dir
        self.verbose = verbose

        # Load model and components
        loaded = SPTClassifierTrainer.load_model(model_dir, verbose=verbose)

        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.feature_extractor = loaded['feature_extractor']
        self.metadata = loaded['metadata']
        self.feature_config = loaded['feature_config']

        self.selected_physics_features = self.metadata.get(
            'selected_physics_features',
            self.feature_config.get('selected_physics_features', SPTClassifierTrainer.SELECTED_PHYSICS_FEATURES)
        )
        self.experimental_features = self.metadata.get(
            'experimental_features',
            self.feature_config.get('experimental_features', SPTClassifierTrainer.EXPERIMENTAL_FEATURES)
        )

        all_physics = self.feature_config.get('all_physics_features', list(self.feature_extractor.feature_names))
        try:
            self.selected_physics_indices = [
                all_physics.index(name) for name in self.selected_physics_features
            ]
        except ValueError as exc:
            missing = [name for name in self.selected_physics_features if name not in all_physics]
            raise ValueError(f"Missing physics features in extractor: {missing}") from exc

        self.class_names = self.metadata['class_names']
        self.max_length = self.metadata['max_length']
        self.input_dim = self.metadata['input_dim']

        if verbose:
            print(f"\nModel ready for prediction!")
            print(f"Classes: {self.class_names}")

    def predict(
        self,
        trajectories: List[np.ndarray],
        D_values: np.ndarray,
        poly_degrees: np.ndarray,
        return_probabilities: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict diffusion type for trajectories

        Args:
            trajectories: List of trajectory arrays (N, d) where d=2 or 3
            D_values: Diffusion coefficients [µm²/s] - shape (len(trajectories),)
            poly_degrees: Polymerization degrees [0-1] - shape (len(trajectories),)
            return_probabilities: If True, also return class probabilities

        Returns:
            predictions: Class indices (0=Normal, 1=Sub, 2=Super, 3=Confined)
            probabilities: (optional) Class probabilities (N, 4)
        """
        n_samples = len(trajectories)

        if len(D_values) != n_samples or len(poly_degrees) != n_samples:
            raise ValueError(f"Mismatch: {n_samples} trajectories but {len(D_values)} D_values and {len(poly_degrees)} poly_degrees")

        if self.verbose:
            print(f"\nPredicting {n_samples} trajectories...")

        # 1. Extract physics-based features
        physics_matrix = self.feature_extractor.extract_batch(trajectories, n_jobs=-1)
        physics_selected = physics_matrix[:, self.selected_physics_indices]

        # 2. Add experimental features (D, Poly, Dim)
        D_log = np.log10(np.clip(D_values, 1e-12, None)).reshape(-1, 1)
        poly_feat = poly_degrees.reshape(-1, 1)
        dim_feat = np.array([0.0 if traj.shape[1] == 2 else 1.0 for traj in trajectories]).reshape(-1, 1)

        X_feat = np.concatenate([physics_selected, D_log, poly_feat, dim_feat], axis=1)

        # 3. Scale features
        X_feat_scaled = self.scaler.transform(X_feat)

        # 4. Normierte Trajektorien vorbereiten
        X_traj_padded = SPTClassifierTrainer.preprocess_trajectories(
            trajectories,
            self.max_length,
            self.input_dim
        )

        # 5. Predict
        probabilities = self.model.predict([X_traj_padded, X_feat_scaled], verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        if self.verbose:
            print(f"\nPredictions complete!")
            for i, class_name in enumerate(self.class_names):
                count = np.sum(predictions == i)
                print(f"  {class_name}: {count} ({100*count/n_samples:.1f}%)")

        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions, None

    def predict_single(
        self,
        trajectory: np.ndarray,
        D_value: float,
        poly_degree: float,
        verbose: bool = True
    ) -> Tuple[str, np.ndarray]:
        """
        Predict single trajectory

        Args:
            trajectory: Single trajectory (N, d)
            D_value: Diffusion coefficient [µm²/s]
            poly_degree: Polymerization degree [0-1]
            verbose: Print result

        Returns:
            class_name: Predicted class name
            probabilities: Class probabilities (4,)
        """
        original_verbose = self.verbose
        self.verbose = False

        predictions, probs = self.predict(
            [trajectory],
            np.array([D_value]),
            np.array([poly_degree]),
            return_probabilities=True
        )

        self.verbose = original_verbose

        class_idx = predictions[0]
        class_name = self.class_names[class_idx]

        if verbose:
            print(f"\nPrediction: {class_name}")
            print(f"Confidence: {probs[0][class_idx]:.1%}")
            print(f"\nAll probabilities:")
            for i, name in enumerate(self.class_names):
                print(f"  {name}: {probs[0][i]:.1%}")

        return class_name, probs[0]


def demo_prediction(model_dir: str):
    """
    Demonstrate prediction on synthetic test data
    """
    print("="*80)
    print("SPT CLASSIFIER - PREDICTION DEMO")
    print("="*80 + "\n")

    # Load model
    predictor = SPTPredictor(model_dir, verbose=True)

    # Generate test trajectories
    print("\n" + "="*80)
    print("GENERATING TEST TRAJECTORIES")
    print("="*80)

    X_test, y_test, lengths_test, class_names, D_test, poly_test = generate_spt_dataset(
        n_samples_per_class=25,
        min_length=50,
        max_length=500,
        dimensionality='2D',
        dt=0.01,
        localization_precision=0.015,
        verbose=True,
        use_full_D_range=True,
        augment_polymerization=True
    )

    # Make predictions
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)

    predictions, probabilities = predictor.predict(
        X_test,
        D_test,
        poly_test,
        return_probabilities=True
    )

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    accuracy = np.mean(predictions == y_test)
    print(f"\nOverall Accuracy: {accuracy:.1%}")

    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == y_test[mask])
            print(f"  {class_name}: {class_acc:.1%} ({np.sum(mask)} samples)")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Add counts
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    output_path = Path(model_dir) / 'prediction_demo_confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved: {output_path}")

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict diffusion types for SPT trajectories')
    parser.add_argument('--model_dir', type=str, default='./spt_trained_model_app',
                        help='Directory containing trained model')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration on synthetic data')

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        demo_prediction(args.model_dir)
    else:
        print("Use --demo to run a demonstration on synthetic data")
        print("\nFor custom predictions, use the SPTPredictor class in your own scripts:")
        print("\n  from predict_trajectories import SPTPredictor")
        print(f"  predictor = SPTPredictor('{args.model_dir}')")
        print("  predictions = predictor.predict(trajectories, D_values, poly_degrees)")
