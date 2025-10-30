"""Fast end-to-end training pipeline for SPT diffusion classification.

This module provides the :class:`SPTClassifierTrainer` which orchestrates
synthetic trajectory generation, physics inspired feature extraction, model
training and persistence.  The goal of the redesign is twofold:

* **Extremely fast feedback** – the defaults are tuned for sub-minute
  experiments on a laptop CPU while still scaling to larger datasets on a GPU.
* **Excellent accuracy** – a light-weight but expressive hybrid network that
  fuses trajectory and feature representations, combined with stratified
  splits, class rebalancing and advanced regularisation.

The trainer exposes simple one-click orchestration for the GUI application and
can also be scripted programmatically.  Artifacts (Keras model, scaler,
feature names, label map, history and metadata) are persisted in a single
output directory so that the model can be reloaded for later evaluation.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from spt_feature_extractor import SPTFeatureExtractor
from spt_trajectory_generator import generate_spt_dataset

ArrayLike = np.ndarray


@dataclass
class TrainingConfig:
    """Container describing how a dataset should be generated and trained."""

    n_samples_per_class: int = 400
    mode: str = "both"  # "2d", "3d" or "both"
    ratio_3d: float = 0.4
    polymerization_degree: float = 0.5
    epochs: int = 40
    batch_size: int = 128
    cache_path: Optional[Path] = None

    def canonical_mode(self) -> str:
        mode = self.mode.strip().lower()
        if mode not in {"2d", "3d", "both"}:
            raise ValueError("mode must be '2D', '3D' or 'both'")
        return mode


@dataclass
class DatasetBundle:
    """Holds the padded trajectory tensors and scaled feature matrices."""

    X_traj_train: ArrayLike
    X_traj_val: ArrayLike
    X_traj_test: ArrayLike
    X_feat_train: ArrayLike
    X_feat_val: ArrayLike
    X_feat_test: ArrayLike
    y_train: ArrayLike
    y_val: ArrayLike
    y_test: ArrayLike
    lengths: ArrayLike
    class_names: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)
    scaler_state: Dict[str, ArrayLike] = field(default_factory=dict)


class EpochProgressCallback(keras.callbacks.Callback):
    """Keras callback that forwards epoch end statistics to the GUI."""

    def __init__(self, total_epochs: int, sink: Optional[Callable[[int, Dict[str, float], int], None]] = None):
        super().__init__()
        self.total_epochs = int(total_epochs)
        self.sink = sink

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:  # noqa: D401
        if self.sink is not None:
            self.sink(epoch, dict(logs or {}), self.total_epochs)


class SPTClassifierTrainer:
    """Fast hybrid CNN-BiLSTM-attention classifier for SPT trajectories."""

    def __init__(
        self,
        max_length: int = 400,
        output_dir: str | Path = "./spt_trained_model",
        random_seed: int = 42,
        enable_mixed_precision: bool = True,
    ) -> None:
        self.max_length = int(max_length)
        self.random_seed = int(random_seed)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._set_random_seeds(self.random_seed)
        self.mixed_precision = False
        if enable_mixed_precision and tf.config.list_physical_devices("GPU"):
            try:
                from tensorflow.keras import mixed_precision

                mixed_precision.set_global_policy("mixed_float16")
                self.mixed_precision = True
            except Exception:  # pragma: no cover - defensive, mixed precision optional
                self.mixed_precision = False

        self.feature_extractor = SPTFeatureExtractor(dt=0.1)
        self.scaler = StandardScaler()

        self.dataset: Optional[DatasetBundle] = None
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self.class_weights: Optional[Dict[int, float]] = None
        self.class_names: List[str] = ["Normal", "Subdiffusion", "Superdiffusion", "Confined"]

    # ------------------------------------------------------------------
    # Dataset generation and preprocessing
    # ------------------------------------------------------------------
    def generate_training_data(
        self,
        config: TrainingConfig | None = None,
        *,
        n_samples_per_class: Optional[int] = None,
        mode: Optional[str] = None,
        ratio_3d: Optional[float] = None,
        polymerization_degree: Optional[float] = None,
        verbose: bool = True,
        reuse_cache: bool = True,
    ) -> None:
        """Create synthetic trajectories, extract features and split the data."""

        config = self._resolve_config(
            config,
            n_samples_per_class=n_samples_per_class,
            mode=mode,
            ratio_3d=ratio_3d,
            polymerization_degree=polymerization_degree,
        )

        cache_path = config.cache_path or (self.output_dir / "cached_dataset.npz")
        use_cache = False
        if reuse_cache and cache_path.is_file():
            if self._cache_matches_config(cache_path, config):
                if verbose:
                    print(f"Loading cached dataset from {cache_path}")
                bundle = self._load_cached_dataset(cache_path)
                use_cache = True
            elif verbose:
                print("Cached dataset configuration mismatch – regenerating …")

        if not use_cache:
            if verbose:
                print("Generating synthetic trajectories …")
            bundle = self._create_dataset(config, verbose=verbose)
            if cache_path:
                self._store_cached_dataset(cache_path, bundle)

        self.dataset = bundle
        self.class_names = bundle.class_names
        self.class_weights = self._compute_class_weights(bundle.y_train)
        self._restore_scaler(bundle.scaler_state)

        if verbose:
            print(
                f"Dataset ready: train={bundle.X_traj_train.shape[0]}, "
                f"val={bundle.X_traj_val.shape[0]}, test={bundle.X_traj_test.shape[0]}"
            )

    # ------------------------------------------------------------------
    # Model building and training
    # ------------------------------------------------------------------
    def build_model(self, *, learning_rate: float = 1e-3, verbose: bool = True) -> keras.Model:
        """Construct the dual-branch neural network."""

        if self.dataset is None:
            raise RuntimeError("Call generate_training_data() before build_model().")

        n_features = self.dataset.X_feat_train.shape[1]
        traj_input = keras.Input(shape=(self.max_length, 3), name="trajectory")
        x = layers.Masking(mask_value=0.0)(traj_input)
        x = layers.SeparableConv1D(64, 5, padding="same", activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv1D(64, 5, padding="same", activation="swish")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.15))(x)
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)
        x = attention(x, x)
        x = layers.GlobalAveragePooling1D()(x)

        feat_input = keras.Input(shape=(n_features,), name="features")
        y = layers.BatchNormalization()(feat_input)
        y = layers.Dense(96, activation="swish")(y)
        y = layers.Dropout(0.2)(y)

        combined = layers.Concatenate()([x, y])
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dense(128, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-5))(combined)
        combined = layers.Dropout(0.25)(combined)

        output = layers.Dense(len(self.class_names), activation="softmax", dtype="float32")(combined)

        model = keras.Model(inputs=[traj_input, feat_input], outputs=output, name="spt_classifier")
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        if verbose:
            model.summary()
        return model

    def train(
        self,
        *,
        epochs: int = 40,
        batch_size: int = 128,
        verbose: int = 1,
        epoch_callback: Optional[Callable[[int, Dict[str, float], int], None]] = None,
    ) -> keras.callbacks.History:
        """Train the model using tf.data pipelines for maximum speed."""

        if self.dataset is None or self.model is None:
            raise RuntimeError("Dataset and model must be prepared before training.")

        train_ds = self._make_dataset(
            self.dataset.X_traj_train,
            self.dataset.X_feat_train,
            self.dataset.y_train,
            batch_size,
            shuffle=True,
        )
        val_ds = self._make_dataset(
            self.dataset.X_traj_val,
            self.dataset.X_feat_val,
            self.dataset.y_val,
            batch_size,
            shuffle=False,
        )

        callbacks = self._build_callbacks(epochs)
        if epoch_callback is not None:
            callbacks.append(EpochProgressCallback(epochs, epoch_callback))

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=self.class_weights,
        )

        self.history = history
        return history

    # ------------------------------------------------------------------
    # Evaluation and persistence
    # ------------------------------------------------------------------
    def evaluate(self, *, verbose: bool = True) -> Tuple[float, Dict[str, Dict[str, float]], np.ndarray]:
        """Evaluate the model on the held-out test set."""

        if self.dataset is None or self.model is None:
            raise RuntimeError("Nothing to evaluate – run generate_training_data() and build_model() first.")

        test_ds = self._make_dataset(
            self.dataset.X_traj_test,
            self.dataset.X_feat_test,
            self.dataset.y_test,
            batch_size=256,
            shuffle=False,
        )
        loss, accuracy = self.model.evaluate(test_ds, verbose=0)

        y_pred_prob = self.model.predict(test_ds, verbose=0)
        y_pred = y_pred_prob.argmax(axis=1)
        report = classification_report(
            self.dataset.y_test,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(self.dataset.y_test, y_pred)

        if verbose:
            print(json.dumps(report, indent=2))
            print("Confusion matrix:\n", matrix)
            print(f"Test accuracy: {accuracy:.4f}")

        return accuracy, report, matrix

    def predict_trajectories(self, trajectories: Iterable[np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """Classify arbitrary trajectories using the trained model."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call from_artifacts() or train a model first.")

        trajectories = list(trajectories)
        if not trajectories:
            return [], np.empty((0, len(self.class_names)))

        traj_tensor = self._pad_and_normalise(trajectories)
        features = self.feature_extractor.extract_batch(trajectories)
        features = self.scaler.transform(features).astype(np.float32)

        probs = self.model.predict([traj_tensor, features], verbose=0)
        labels = [self.class_names[int(idx)] for idx in probs.argmax(axis=1)]
        return labels, probs

    def save_model(self, *, verbose: bool = True) -> Path:
        """Persist the trained model and preprocessing artifacts."""

        if self.model is None or self.dataset is None:
            raise RuntimeError("Train a model before calling save_model().")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_dir = self.output_dir / f"run_{timestamp}"
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "model.keras"
        self.model.save(model_path, include_optimizer=True)

        scaler_path = target_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        feature_path = target_dir / "feature_names.json"
        with open(feature_path, "w", encoding="utf-8") as f:
            json.dump(self.feature_extractor.feature_names, f, indent=2)

        labels_path = target_dir / "class_names.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(self.class_names, f, indent=2)

        history_path = target_dir / "history.json"
        if self.history is not None:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history.history, f, indent=2)

        dataset_meta = self.dataset.metadata | {
            "max_length": self.max_length,
            "input_channels": 3,
            "class_weights": self.class_weights,
            "class_names": self.class_names,
        }
        metadata_path = target_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset_meta, f, indent=2, default=_json_converter)

        if verbose:
            print(f"Artifacts saved to {target_dir}")

        return target_dir

    # ------------------------------------------------------------------
    # Convenience orchestration
    # ------------------------------------------------------------------
    def run_complete_training(self, config: TrainingConfig, *, verbose: bool = True) -> Path:
        """Convenience wrapper for command line and GUI one-click training."""

        if verbose:
            print("=== SPT classifier one-click training ===")
            print(asdict(config))

        self.generate_training_data(config=config, verbose=verbose)
        self.build_model(verbose=verbose)
        self.train(epochs=config.epochs, batch_size=config.batch_size, verbose=1 if verbose else 0)
        self.evaluate(verbose=verbose)
        return self.save_model(verbose=verbose)

    # ------------------------------------------------------------------
    # Loading previously trained models
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(cls, directory: str | Path) -> "SPTClassifierTrainer":
        """Restore a trainer with model and scaler for inference."""

        directory = Path(directory)
        metadata_file = directory / "metadata.json"
        if not metadata_file.is_file():
            raise FileNotFoundError(f"metadata.json missing in {directory}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        trainer = cls(max_length=int(metadata.get("max_length", 400)), output_dir=directory)
        trainer.class_names = metadata.get("class_names", trainer.class_names)

        scaler_path = directory / "feature_scaler.pkl"
        with open(scaler_path, "rb") as f:
            trainer.scaler = pickle.load(f)

        model_path = directory / "model.keras"
        trainer.model = keras.models.load_model(model_path)

        trainer.dataset = None
        trainer.history = None
        return trainer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_config(
        self,
        config: Optional[TrainingConfig],
        *,
        n_samples_per_class: Optional[int],
        mode: Optional[str],
        ratio_3d: Optional[float],
        polymerization_degree: Optional[float],
    ) -> TrainingConfig:
        if config is None:
            config = TrainingConfig()
        if n_samples_per_class is not None:
            config.n_samples_per_class = int(n_samples_per_class)
        if mode is not None:
            config.mode = mode
        if ratio_3d is not None:
            config.ratio_3d = float(np.clip(ratio_3d, 0.0, 1.0))
        if polymerization_degree is not None:
            config.polymerization_degree = float(np.clip(polymerization_degree, 0.0, 1.0))
        config.canonical_mode()  # validates mode
        return config

    def _config_signature(self, config: TrainingConfig) -> Dict[str, object]:
        mode = config.canonical_mode()
        return {
            "mode": mode,
            "ratio_3d": round(float(config.ratio_3d), 6),
            "polymerization_degree": round(float(config.polymerization_degree), 6),
            "n_samples_per_class": int(config.n_samples_per_class),
            "max_length": int(self.max_length),
            "feature_dim": int(len(self.feature_extractor.feature_names)),
        }

    def _create_dataset(self, config: TrainingConfig, *, verbose: bool) -> DatasetBundle:
        plan = self._plan_generation(config)
        all_traj: List[np.ndarray] = []
        labels: List[int] = []
        lengths: List[int] = []
        class_names: List[str] | None = None

        for dim_label, count in plan:
            X_dim, y_dim, length_dim, class_names_dim = generate_spt_dataset(
                n_samples_per_class=count,
                min_length=80,
                max_length=self.max_length,
                dimensionality=dim_label,
                polymerization_degree=config.polymerization_degree,
                dt=0.1,
                localization_precision=0.018,
                boost_classes=None,
                verbose=verbose,
            )
            all_traj.extend(X_dim)
            labels.append(y_dim)
            lengths.append(length_dim)
            if class_names is None:
                class_names = list(class_names_dim)

        y = np.concatenate(labels)
        lengths_array = np.concatenate(lengths)
        class_names = class_names or self.class_names

        X_traj = self._pad_and_normalise(all_traj)
        X_feat = self.feature_extractor.extract_batch(all_traj)

        (X_traj_train, X_traj_temp, X_feat_train, X_feat_temp, y_train, y_temp) = train_test_split(
            X_traj,
            X_feat,
            y,
            test_size=0.3,
            random_state=self.random_seed,
            stratify=y,
        )
        (X_traj_val, X_traj_test, X_feat_val, X_feat_test, y_val, y_test) = train_test_split(
            X_traj_temp,
            X_feat_temp,
            y_temp,
            test_size=0.5,
            random_state=self.random_seed,
            stratify=y_temp,
        )

        self.scaler.fit(X_feat_train)
        X_feat_train = self.scaler.transform(X_feat_train)
        X_feat_val = self.scaler.transform(X_feat_val)
        X_feat_test = self.scaler.transform(X_feat_test)

        bundle = DatasetBundle(
            X_traj_train=X_traj_train,
            X_traj_val=X_traj_val,
            X_traj_test=X_traj_test,
            X_feat_train=X_feat_train.astype(np.float32),
            X_feat_val=X_feat_val.astype(np.float32),
            X_feat_test=X_feat_test.astype(np.float32),
            y_train=y_train.astype(int),
            y_val=y_val.astype(int),
            y_test=y_test.astype(int),
            lengths=lengths_array,
            class_names=class_names,
            metadata={
                "polymerization_degree": float(config.polymerization_degree),
                "ratio_3d": float(config.ratio_3d),
                "mode": config.canonical_mode(),
                "n_samples_per_class": int(config.n_samples_per_class),
                "max_length": int(self.max_length),
                "config_signature": self._config_signature(config),
                "feature_names": list(self.feature_extractor.feature_names),
                "class_names": class_names,
                "total_samples": int(len(y)),
            },
            scaler_state=self._capture_scaler_state(),
        )
        return bundle

    def _plan_generation(self, config: TrainingConfig) -> List[Tuple[str, int]]:
        mode = config.canonical_mode()
        samples = max(1, int(config.n_samples_per_class))
        if mode == "2d":
            return [("2D", samples)]
        if mode == "3d":
            return [("3D", samples)]
        n_3d = int(round(samples * config.ratio_3d))
        n_3d = np.clip(n_3d, 0, samples)
        n_2d = samples - n_3d
        plan: List[Tuple[str, int]] = []
        if n_2d > 0:
            plan.append(("2D", n_2d))
        if n_3d > 0:
            plan.append(("3D", n_3d))
        return plan or [("2D", samples)]

    def _pad_and_normalise(self, trajectories: Iterable[np.ndarray]) -> np.ndarray:
        padded = np.zeros((len(trajectories), self.max_length, 3), dtype=np.float32)
        for i, traj in enumerate(trajectories):
            traj = np.asarray(traj, dtype=np.float32)
            if traj.ndim == 1:
                traj = traj[:, None]
            if traj.shape[1] == 2:
                traj = np.concatenate([traj, np.zeros((traj.shape[0], 1), dtype=np.float32)], axis=1)
            elif traj.shape[1] > 3:
                traj = traj[:, :3]

            centred = traj - traj[0:1]
            scale = np.std(centred) + 1e-6
            centred /= scale

            length = min(self.max_length, centred.shape[0])
            if centred.shape[0] >= self.max_length:
                start = np.random.randint(0, centred.shape[0] - self.max_length + 1)
                segment = centred[start : start + self.max_length]
                padded[i] = segment
            else:
                padded[i, :length, :] = centred[:length]
        return padded

    def _compute_class_weights(self, y_train: ArrayLike) -> Optional[Dict[int, float]]:
        classes = np.unique(y_train)
        if len(classes) < 2:
            return None
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return {int(cls): float(w) for cls, w in zip(classes, weights)}

    def _make_dataset(
        self,
        X_traj: ArrayLike,
        X_feat: ArrayLike,
        y: ArrayLike,
        batch_size: int,
        *,
        shuffle: bool,
    ) -> tf.data.Dataset:
        y_onehot = tf.one_hot(y.astype(int), depth=len(self.class_names), dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices(((X_traj, X_feat), y_onehot))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X_traj), seed=self.random_seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    def _build_callbacks(self, epochs: int) -> List[keras.callbacks.Callback]:
        early = keras.callbacks.EarlyStopping(patience=max(5, epochs // 5), restore_best_weights=True, monitor="val_accuracy")
        reduce = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(3, epochs // 6), monitor="val_loss", min_lr=1e-5)
        return [early, reduce]

    def _cache_matches_config(self, path: Path, config: TrainingConfig) -> bool:
        try:
            with np.load(path, allow_pickle=True) as data:
                metadata_raw = data.get("metadata")
                if metadata_raw is None:
                    return False
                if hasattr(metadata_raw, "item"):
                    metadata_raw = metadata_raw.item()
                metadata = json.loads(str(metadata_raw))
        except Exception:
            return False

        stored_signature = metadata.get("config_signature")
        if not isinstance(stored_signature, dict):
            return False
        expected = self._config_signature(config)
        return self._signatures_match(stored_signature, expected)

    def _store_cached_dataset(self, path: Path, bundle: DatasetBundle) -> None:
        payload = {
            "X_traj_train": bundle.X_traj_train,
            "X_traj_val": bundle.X_traj_val,
            "X_traj_test": bundle.X_traj_test,
            "X_feat_train": bundle.X_feat_train,
            "X_feat_val": bundle.X_feat_val,
            "X_feat_test": bundle.X_feat_test,
            "y_train": bundle.y_train,
            "y_val": bundle.y_val,
            "y_test": bundle.y_test,
            "lengths": bundle.lengths,
            "class_names": np.array(bundle.class_names, dtype=object),
            "metadata": json.dumps(bundle.metadata),
        }
        if bundle.scaler_state:
            if bundle.scaler_state.get("mean") is not None:
                payload["scaler_mean"] = np.asarray(bundle.scaler_state["mean"], dtype=np.float64)
            if bundle.scaler_state.get("scale") is not None:
                payload["scaler_scale"] = np.asarray(bundle.scaler_state["scale"], dtype=np.float64)
            if bundle.scaler_state.get("var") is not None:
                payload["scaler_var"] = np.asarray(bundle.scaler_state["var"], dtype=np.float64)
            if bundle.scaler_state.get("n_features") is not None:
                payload["scaler_n_features"] = np.asarray(bundle.scaler_state["n_features"], dtype=np.int64)
            if bundle.scaler_state.get("n_samples") is not None:
                payload["scaler_n_samples"] = np.asarray(bundle.scaler_state["n_samples"], dtype=np.int64)
            if "feature_names" in bundle.scaler_state:
                payload["scaler_feature_names"] = np.asarray(bundle.scaler_state["feature_names"], dtype=object)

        np.savez_compressed(path, **payload)

    def _load_cached_dataset(self, path: Path) -> DatasetBundle:
        with np.load(path, allow_pickle=True) as data:
            metadata_raw = data["metadata"]
            if hasattr(metadata_raw, "item"):
                metadata_raw = metadata_raw.item()
            metadata = json.loads(str(metadata_raw))
            scaler_state: Dict[str, ArrayLike] = {}
            if "scaler_mean" in data:
                scaler_state["mean"] = data["scaler_mean"]
            if "scaler_scale" in data:
                scaler_state["scale"] = data["scaler_scale"]
            if "scaler_var" in data:
                scaler_state["var"] = data["scaler_var"]
            if "scaler_n_features" in data:
                scaler_state["n_features"] = data["scaler_n_features"]
            if "scaler_n_samples" in data:
                scaler_state["n_samples"] = data["scaler_n_samples"]
            if "scaler_feature_names" in data:
                scaler_state["feature_names"] = data["scaler_feature_names"]
            return DatasetBundle(
                X_traj_train=data["X_traj_train"],
                X_traj_val=data["X_traj_val"],
                X_traj_test=data["X_traj_test"],
                X_feat_train=data["X_feat_train"],
                X_feat_val=data["X_feat_val"],
                X_feat_test=data["X_feat_test"],
                y_train=data["y_train"],
                y_val=data["y_val"],
                y_test=data["y_test"],
                lengths=data["lengths"],
                class_names=list(data["class_names"].tolist()),
                metadata=metadata,
                scaler_state=scaler_state,
            )

    def _capture_scaler_state(self) -> Dict[str, ArrayLike]:
        state: Dict[str, ArrayLike] = {
            "mean": np.asarray(self.scaler.mean_, dtype=np.float64),
            "scale": np.asarray(self.scaler.scale_, dtype=np.float64),
            "var": np.asarray(self.scaler.var_, dtype=np.float64),
            "n_features": np.asarray([self.scaler.n_features_in_], dtype=np.int64),
            "n_samples": np.asarray([int(getattr(self.scaler, "n_samples_seen_", 0))], dtype=np.int64),
        }
        if hasattr(self.scaler, "feature_names_in_"):
            state["feature_names"] = np.asarray(self.scaler.feature_names_in_, dtype=object)
        return state

    def _restore_scaler(self, scaler_state: Dict[str, ArrayLike]) -> None:
        if not scaler_state:
            return

        mean = scaler_state.get("mean")
        scale = scaler_state.get("scale")
        var = scaler_state.get("var")
        if mean is None or scale is None or var is None:
            return

        self.scaler.mean_ = np.asarray(mean, dtype=np.float64)
        self.scaler.scale_ = np.asarray(scale, dtype=np.float64)
        self.scaler.var_ = np.asarray(var, dtype=np.float64)
        n_features = scaler_state.get("n_features")
        if n_features is not None:
            self.scaler.n_features_in_ = int(np.asarray(n_features).ravel()[0])
        else:
            self.scaler.n_features_in_ = self.scaler.mean_.shape[0]
        n_samples = scaler_state.get("n_samples")
        if n_samples is not None:
            seen = int(np.asarray(n_samples).ravel()[0])
            self.scaler.n_samples_seen_ = max(1, seen)
        else:
            self.scaler.n_samples_seen_ = max(1, self.scaler.mean_.shape[0])

        feature_names = scaler_state.get("feature_names")
        if feature_names is not None:
            names = [str(x) for x in np.asarray(feature_names).ravel()]
            if names:
                self.scaler.feature_names_in_ = np.asarray(names, dtype=object)

    @staticmethod
    def _signatures_match(stored: Dict[str, object], expected: Dict[str, object], tol: float = 1e-6) -> bool:
        for key, expected_value in expected.items():
            if key not in stored:
                return False
            value = stored[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(expected_value, float):
                try:
                    if abs(float(value) - expected_value) > tol:
                        return False
                except (TypeError, ValueError):
                    return False
            else:
                if value != expected_value:
                    return False
        return True

    @staticmethod
    def _set_random_seeds(seed: int) -> None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def _json_converter(obj):  # pragma: no cover - fallback for numpy types
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


if __name__ == "__main__":
    trainer = SPTClassifierTrainer(max_length=400)
    config = TrainingConfig(n_samples_per_class=150, mode="both", ratio_3d=0.35, epochs=25, batch_size=128)
    artifacts = trainer.run_complete_training(config)
    print(f"Training run complete. Artifacts stored at {artifacts}")
