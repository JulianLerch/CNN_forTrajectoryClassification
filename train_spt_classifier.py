# -*- coding: utf-8 -*-
"""
SPT CLASSIFIER - KOMPLETTES TRAINING SYSTEM
============================================

Wissenschaftlich fundiertes End-to-End Training System fÃ¼r Single-Particle-Tracking 
Diffusionsklassifikation mit Deep Learning.

ARCHITEKTUR: Hybrid CNN-LSTM mit Multi-Head Attention
- 1D Convolutional Layers (lokale Muster)
- Bidirectional LSTM (temporale AbhÃ¤ngigkeiten)
- Multi-Head Self-Attention (wichtige Zeitpunkte)
- Feature Branch (24 physikalische Features)

WISSENSCHAFTLICHE BASIS:
- Granik & Weiss (2019): Deep Learning fÃ¼r SPT-Klassifikation
- Hochreiter & Schmidhuber (1997): LSTM fÃ¼r Sequenzverarbeitung
- Vaswani et al. (2017): Attention Mechanisms

Autor: Masterthesis TDI-G0 Diffusion in Polymeren
Version: 2.0 - VollstÃ¤ndig korrigiert und integriert
Oktober 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Lokale Module
from spt_trajectory_generator import generate_spt_dataset, PolymerDiffusionParameters
from spt_feature_extractor import SPTFeatureExtractor

print("="*80)
print("SPT DEEP LEARNING CLASSIFIER - TRAINING SYSTEM")
print("="*80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print("="*80 + "\n")


class SPTClassifierTrainer:
    """
    VollstÃ¤ndiges Training System fÃ¼r SPT-Klassifikation
    
    Integriert:
    -----------
    1. Datengenerierung (physikalisch korrekte synthetische Trajektorien)
    2. Feature-Extraktion (24 wissenschaftliche Features)
    3. Deep Learning Architektur (Hybrid CNN-LSTM-Attention)
    4. Training mit Best Practices (Early Stopping, LR Scheduling)
    5. Evaluation & Visualisierung
    6. Model Persistence
    
    Theoretische Fundierung:
    -------------------------
    Die Architektur kombiniert:
    - **Convolutional Layers**: Erfassen lokale Muster in Trajektorien
      (z.B. kurzfristige subdiffusive Phasen)
    - **LSTM**: Modelliert langreichweitige temporale AbhÃ¤ngigkeiten
      und verarbeitet variable Trajektorien-LÃ¤ngen
    - **Attention**: Identifiziert informative Abschnitte der Trajektorie
    - **Feature Branch**: Inkorporiert physikalisches DomÃ¤nenwissen
      (Physics-informed Machine Learning)
    """
    
    def __init__(
        self,
        max_length: int = 500,
        output_dir: str = './spt_trained_model',
        random_seed: int = 42
    ):
        """
        Args:
            max_length: Maximale Trajektorien-LÃ¤nge fÃ¼r Padding
            n_features: Anzahl extrahierter Features (24)
            output_dir: Ausgabe-Verzeichnis fÃ¼r trainiertes Modell
            random_seed: Random Seed fÃ¼r Reproduzierbarkeit
        """
        self.max_length = max_length
        self.n_features = None
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        # Erstelle Output-Verzeichnis
        os.makedirs(output_dir, exist_ok=True)
        
        # Set Random Seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Komponenten
        self.feature_extractor = SPTFeatureExtractor(dt=0.01)
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        
        # Daten
        self.X_traj_train = None
        self.X_feat_train = None
        self.y_train = None
        self.X_traj_val = None
        self.X_feat_val = None
        self.y_val = None
        self.X_traj_test = None
        self.X_feat_test = None
        self.y_test = None
        self.class_names = ['Normal', 'Subdiffusion', 'Superdiffusion', 'Confined']
        # Einheitliche Eingabe-Dimension: 3 (2D wird mit z=0 aufgefÃ¼llt)
        self.input_dim = 3
    
    def generate_training_data(
        self,
        n_samples_per_class: int = 2000,
        mode: str = 'both',
        ratio_3d: float = 0.5,
        polymerization_degree: float = 0.5,
        verbose: bool = True
    ):
        """
        Generiere wissenschaftlich fundierte Trainingsdaten
        
        Args:
            n_samples_per_class: Anzahl Samples pro Klasse
            mode: '2D', '3D' oder 'both'
            ratio_3d: Anteil 3D-Samples bei 'both' (0..1)
            polymerization_degree: 0.0-1.0 (Eduktschmelze bis vollpolymerisiert)
            verbose: Ausgabe?
            
        Wissenschaftliche Basis:
        ------------------------
        Die Datengenerierung basiert auf physikalisch korrekten Modellen:
        
        1. **Normale Diffusion**: Einstein-Smoluchowski Gleichung
           MSD(t) = 2dÂ·DÂ·t (d=DimensionalitÃ¤t)
           
        2. **Anomale Diffusion**: Generalisierte Diffusionsgleichung
           MSD(t) = 2dÂ·D_Î±Â·t^Î±
           - Î± < 1: Subdiffusion (Polymer-KÃ¤fige, Crowding)
           - Î± > 1: Superdiffusion (aktive Prozesse, Konvektion)
           
        3. **Confined Diffusion**: Harmonisches Potential
           Reflektive Randbedingungen in sphÃ¤rischer/zylindrischer Geometrie
           
        4. **Lokalisierungs-PrÃ¤zision**: GauÃŸsches Rauschen
           Ïƒ_loc â‰ˆ 15 nm (typisch fÃ¼r TIRF-Mikroskopie)
        """
        n_samples_per_class = int(n_samples_per_class)
        if n_samples_per_class <= 0:
            raise ValueError("n_samples_per_class must be a positive integer")

        mode_normalized = mode.lower()
        if mode_normalized not in {"2d", "3d", "both"}:
            raise ValueError("mode must be one of '2D', '3D' or 'both'")

        ratio_3d = float(np.clip(ratio_3d, 0.0, 1.0))

        if mode_normalized == "2d":
            generation_plan = [("2D", n_samples_per_class)]
            self.input_dim = 2
        elif mode_normalized == "3d":
            generation_plan = [("3D", n_samples_per_class)]
            self.input_dim = 3
        else:
            n_total = max(1, int(n_samples_per_class))
            n_3d = int(round(n_total * ratio_3d))
            n_3d = min(n_total, n_3d)
            if ratio_3d > 0.0 and n_3d == 0:
                n_3d = 1
            n_2d = n_total - n_3d
            if ratio_3d < 1.0 and n_2d == 0:
                n_2d = 1
                n_3d = max(0, n_total - 1)

            generation_plan = []
            if n_2d > 0:
                generation_plan.append(("2D", n_2d))
            if n_3d > 0:
                generation_plan.append(("3D", n_3d))
            self.input_dim = 3

        D_min, D_max = PolymerDiffusionParameters.get_D_range(polymerization_degree)

        if verbose:
            print("\n" + "="*80)
            print("DATENGENERIERUNG")
            print("="*80)
            print(f"Modus: {mode_normalized.upper()}")
            if mode_normalized == "both":
                plan_str = ", ".join([f"{dim}:{count}" for dim, count in generation_plan])
                print(f"Aufteilung 2D/3D (pro Klasse): {plan_str}")
            else:
                print(f"DimensionalitÃ¤t: {generation_plan[0][0]}")
            print(f"Samples pro Klasse: {n_samples_per_class}")
            print(f"Polymerisationsgrad: {polymerization_degree:.1%}")
            print(f"Diffusionskoeffizient-Bereich: {D_min:.2e} - {D_max:.2e} ÂµmÂ²/s")
            print("="*80 + "\n")

        X_traj_list = []
        y_parts = []
        lengths_parts = []
        D_values_parts = []
        poly_degrees_parts = []
        class_names = None

        for dim_label, n_dim_samples in generation_plan:
            X_dim, y_dim, lengths_dim, class_names_dim, D_dim, poly_dim = generate_spt_dataset(
                n_samples_per_class=n_dim_samples,
                min_length=50,
                max_length=self.max_length,
                dimensionality=dim_label,
                polymerization_degree=polymerization_degree,
                dt=0.01,
                localization_precision=0.015,
                boost_classes=None,
                verbose=verbose,
                use_full_D_range=True,
                augment_polymerization=True
            )

            X_traj_list.extend(X_dim)
            y_parts.append(y_dim)
            lengths_parts.append(lengths_dim)
            D_values_parts.append(D_dim)
            poly_degrees_parts.append(poly_dim)

            if class_names is None:
                class_names = class_names_dim

        y = np.concatenate(y_parts)
        lengths = np.concatenate(lengths_parts)
        D_values = np.concatenate(D_values_parts)
        poly_degrees = np.concatenate(poly_degrees_parts)

        self.class_names = class_names
        
        if verbose:
            print(f"\nâœ… {len(X_traj_list)} Trajektorien generiert")
            print(f"   Klassen: {class_names}")
            print(f"   LÃ¤ngen: {lengths.min()}-{lengths.max()} Frames")
        
        # Feature-Extraktion
        if verbose:
            print("\n" + "="*80)
            print("FEATURE-EXTRAKTION")
            print("="*80)
            print("Extrahiere 24 wissenschaftliche Features:")
            print("  â€¢ MSD-basiert (Î±, D_eff, LinearitÃ¤t)")
            print("  â€¢ Geometrisch (Gyration, Asphericity, Straightness)")
            print("  â€¢ Statistisch (Gaussianity, Kurtosis, Skewness)")
            print("  â€¢ Temporal (Velocity Autocorrelation)")
            print("  â€¢ Confinement (Trappedness, Radius Ratio)")
            print("="*80 + "\n")
        
        X_feat = self.feature_extractor.extract_batch(X_traj_list)

        # Füge D-Werte, Polymerisierungsgrad und Dimensionalität als zusätzliche Features hinzu
        # Diese erweitern die 24 physikalischen Features
        D_log = np.log10(D_values).reshape(-1, 1)  # Log-transformiert für bessere Skalierung
        poly_feat = poly_degrees.reshape(-1, 1)

        # Dimensionalität als Feature: 0 für 2D, 1 für 3D
        dim_feat = np.array([0.0 if traj.shape[1] == 2 else 1.0 for traj in X_traj_list]).reshape(-1, 1)

        # Kombiniere alle Features
        X_feat = np.concatenate([X_feat, D_log, poly_feat, dim_feat], axis=1)

        self.n_features = X_feat.shape[1]
        if verbose:
            print(f"\nâœ… Feature-Matrix: {X_feat.shape}")
            print(f"   Basis-Features: 24")
            print(f"   + D-Wert (log10): 1")
            print(f"   + Polymerisierungsgrad: 1")
            print(f"   + Dimensionalität (2D/3D): 1")
            print(f"   = Gesamt: {X_feat.shape[1]} Features pro Sample")
        
        # Padding der Trajektorien
        if verbose:
            print("\n" + "="*80)
            print("TRAJEKTORIEN-PADDING")
            print("="*80)
            print(f"Padding auf max_length = {self.max_length} Frames")
            print("="*80 + "\n")
        
        n_samples = len(X_traj_list)
        X_traj_padded = np.zeros((n_samples, self.max_length, self.input_dim), dtype=np.float32)
        
        for i, traj in enumerate(X_traj_list):
            L = len(traj)
            if L > self.max_length:
                start = np.random.randint(0, L - self.max_length + 1)
                segment = traj[start:start + self.max_length]
            else:
                segment = traj[:L]

            segment = np.asarray(segment, dtype=np.float32)
            if segment.ndim == 1:
                segment = segment[:, np.newaxis]

            if segment.shape[1] < self.input_dim:
                padded = np.zeros((segment.shape[0], self.input_dim), dtype=np.float32)
                padded[:, :segment.shape[1]] = segment
                segment = padded
            elif segment.shape[1] > self.input_dim:
                segment = segment[:, :self.input_dim]

            length = min(segment.shape[0], self.max_length)
            X_traj_padded[i, :length, :] = segment[:length]

        if verbose:
            print(f"âœ… Padded Trajektorien: {X_traj_padded.shape}")
        
        # Train-Val-Test Split (stratifiziert)
        if verbose:
            print("\n" + "="*80)
            print("TRAIN-VAL-TEST SPLIT")
            print("="*80)
            print("Stratifiziert nach Klasse (erhÃ¤lt Klassenverteilung)")
            print("="*80 + "\n")
        
        y_array = np.asarray(y)
        n_total = len(y_array)
        class_counts = np.bincount(y_array.astype(int))
        n_classes = len(class_counts)
        stratify_test = y_array if np.all(class_counts >= 2) else None

        test_size = int(round(n_total * 0.15))
        if stratify_test is not None:
            test_size = max(test_size, n_classes)
        test_size = max(1, min(test_size, n_total - 1))
        if stratify_test is not None and ((test_size < n_classes) or (n_total - test_size < n_classes)):
            stratify_test = None

        X_traj_temp, X_traj_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
            X_traj_padded, X_feat, y,
            test_size=test_size,
            stratify=stratify_test,
            random_state=self.random_seed
        )

        y_temp_array = np.asarray(y_temp)
        stratify_val = None
        if stratify_test is not None:
            temp_counts = np.bincount(y_temp_array.astype(int))
            if np.all(temp_counts >= 2):
                stratify_val = y_temp

        val_size = int(round(len(y_temp) * 0.176))
        if stratify_val is not None:
            val_size = max(val_size, n_classes)
        val_size = max(1, min(val_size, len(y_temp) - 1))
        if stratify_val is not None and ((val_size < n_classes) or (len(y_temp) - val_size < n_classes)):
            stratify_val = None

        X_traj_train, X_traj_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
            X_traj_temp, X_feat_temp, y_temp,
            test_size=val_size,
            stratify=stratify_val,
            random_state=self.random_seed
        )

        if verbose:
            print(f"Train-Set: {len(X_traj_train)} samples ({len(X_traj_train)/len(X_traj_padded)*100:.1f}%)")
            print(f"Val-Set:   {len(X_traj_val)} samples ({len(X_traj_val)/len(X_traj_padded)*100:.1f}%)")
            print(f"Test-Set:  {len(X_traj_test)} samples ({len(X_traj_test)/len(X_traj_padded)*100:.1f}%)")
        
        # Feature-Standardisierung (nur auf Train-Set fitten!)
        if verbose:
            print("\n" + "="*80)
            print("FEATURE-STANDARDISIERUNG")
            print("="*80)
            print("Z-Score Normalisierung: (x - Î¼) / Ïƒ")
            print("Scaler wird nur auf Train-Set gefittet!")
            print("="*80 + "\n")
        
        X_feat_train = self.scaler.fit_transform(X_feat_train)
        X_feat_val = self.scaler.transform(X_feat_val)
        X_feat_test = self.scaler.transform(X_feat_test)
        
        if verbose:
            print("âœ… Features standardisiert")
            print(f"   Train Mean: {np.mean(X_feat_train):.4f}, Std: {np.std(X_feat_train):.4f}")
            print(f"   Val Mean:   {np.mean(X_feat_val):.4f}, Std: {np.std(X_feat_val):.4f}")
        
        # Speichern
        self.X_traj_train = X_traj_train
        self.X_feat_train = X_feat_train
        self.y_train = y_train
        self.X_traj_val = X_traj_val
        self.X_feat_val = X_feat_val
        self.y_val = y_val
        self.X_traj_test = X_traj_test
        self.X_feat_test = X_feat_test
        self.y_test = y_test
        
        if verbose:
            print("\nâœ… Daten erfolgreich vorbereitet!")
    
    def build_model(self, verbose: bool = True):
        """
        Konstruiere Hybrid CNN-LSTM-Attention Architektur
        
        Architektur-Design:
        -------------------
        
        TRAJECTORY BRANCH (Rohdaten):
        Input â†’ Masking â†’ Conv1D Blocks â†’ BiLSTM â†’ Attention â†’ GAP â†’ Dropout
        
        FEATURE BRANCH (24 physikalische Features):
        Input â†’ Dense(128) â†’ BN â†’ ReLU â†’ Dense(64) â†’ BN â†’ ReLU â†’ Dropout
        
        FUSION & CLASSIFICATION:
        Concat(Traj, Feat) â†’ Dense(256) â†’ Dense(128) â†’ Output(4, softmax)
        
        Mathematische Details:
        ----------------------
        
        1. **Conv1D**: Lokale Musterextraktion
           y[n] = Ïƒ(Î£_k w[k]Â·x[n-k] + b)
           Filter: [64, 128, 256], Kernel: [7, 5, 3]
        
        2. **Bidirectional LSTM**: Temporale Sequenzmodellierung
           h_t = LSTM_forward(x_t, h_{t-1})
           h'_t = LSTM_backward(x_t, h'_{t+1})
           output_t = [h_t, h'_t]
        
        3. **Multi-Head Attention**: Wichtige Zeitpunkte identifizieren
           Attention(Q,K,V) = softmax(QK^T / âˆšd_k)Â·V
           Heads: 4
        
        4. **Regularisierung**:
           - Batch Normalization (reduziert Internal Covariate Shift)
           - Dropout: p=0.3 (verhindert Co-Adaptation)
           - L2 Weight Decay: Î»=1e-4
        """
        if verbose:
            print("\n" + "="*80)
            print("MODELL-KONSTRUKTION")
            print("="*80)
            print("Hybrid Architecture: CNN + BiLSTM + Attention + Features")
            print("="*80 + "\n")
        
        # === TRAJECTORY BRANCH ===
        traj_input = layers.Input(
            shape=(self.max_length, self.input_dim),
            name='trajectory_input'
        )
        
        # Masking (ignoriert gepaddete Nullen)
        x_traj = layers.Masking(mask_value=0.0, name='masking')(traj_input)
        
        # Conv1D Blocks
        conv_filters = [64, 128, 256]
        conv_kernels = [7, 5, 3]
        
        for i, (filters, kernel) in enumerate(zip(conv_filters, conv_kernels)):
            x_traj = layers.Conv1D(
                filters=filters,
                kernel_size=kernel,
                padding='same',
                kernel_regularizer=l2(1e-4),
                name=f'conv1d_{i+1}'
            )(x_traj)
            x_traj = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x_traj)
            x_traj = layers.Activation('relu', name=f'relu_conv_{i+1}')(x_traj)
            
            if i < 2:  # MaxPooling nach ersten beiden Blocks
                x_traj = layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x_traj)
        
        # Bidirectional LSTM
        x_traj = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
            name='bilstm'
        )(x_traj)
        
        # Multi-Head Self-Attention
        attention_out = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=128,
            name='attention'
        )(x_traj, x_traj)
        
        # Residual + LayerNorm
        x_traj = layers.Add(name='attention_residual')([x_traj, attention_out])
        x_traj = layers.LayerNormalization(name='attention_norm')(x_traj)
        
        # Global Average Pooling
        x_traj = layers.GlobalAveragePooling1D(name='gap')(x_traj)
        x_traj = layers.Dropout(0.3, name='traj_dropout')(x_traj)
        
        # === FEATURE BRANCH ===
        feat_input = layers.Input(shape=(self.n_features,), name='feature_input')
        
        x_feat = layers.Dense(128, kernel_regularizer=l2(1e-4), name='feat_dense1')(feat_input)
        x_feat = layers.BatchNormalization(name='bn_feat1')(x_feat)
        x_feat = layers.Activation('relu', name='relu_feat1')(x_feat)
        
        x_feat = layers.Dense(64, kernel_regularizer=l2(1e-4), name='feat_dense2')(x_feat)
        x_feat = layers.BatchNormalization(name='bn_feat2')(x_feat)
        x_feat = layers.Activation('relu', name='relu_feat2')(x_feat)
        
        x_feat = layers.Dropout(0.3, name='feat_dropout')(x_feat)
        
        # === FUSION ===
        merged = layers.Concatenate(name='fusion')([x_traj, x_feat])
        
        # Classification Head
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4), name='fusion_dense1')(merged)
        x = layers.BatchNormalization(name='bn_fusion1')(x)
        x = layers.Dropout(0.4, name='fusion_dropout1')(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name='fusion_dense2')(x)
        x = layers.BatchNormalization(name='bn_fusion2')(x)
        x = layers.Dropout(0.4, name='fusion_dropout2')(x)
        
        # Output Layer
        outputs = layers.Dense(4, activation='softmax', name='output')(x)
        
        # Build Model
        model = models.Model(
            inputs=[traj_input, feat_input],
            outputs=outputs,
            name='SPT_Hybrid_Classifier'
        )
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        if verbose:
            print("âœ… Modell konstruiert und kompiliert")
            print(f"\nParameter: {model.count_params():,}")
            print("\nModel Summary:")
            model.summary()
    
    def generate_training_data_both(
        self,
        n_samples_per_class: int = 1000,
        ratio_3d: float = 0.5,
        polymerization_degree: float = 0.5,
        verbose: bool = True
    ):
        """Generate new trajectories for the weakest class and add them to the training set.\n        - target_class_index: 0=Normal, 1=Subdiffusion, 2=Superdiffusion, 3=Confined\n        - n_new: number of new samples across 2D+3D\n        - ratio_3d: share of 3D (0..1). Rest is 2D.\n        """
        if verbose:
            print("\n" + "="*80)
            print("DATENGENERIERUNG (2D+3D)")
            print("="*80)
            print(f"Samples pro Klasse: {n_samples_per_class}")
            print(f"Anteil 3D: {ratio_3d:.0%}")
            print(f"Polymerisationsgrad: {polymerization_degree:.1%}")
            print("="*80 + "\n")

        n_3d = int(round(n_samples_per_class * ratio_3d))
        n_2d = n_samples_per_class - n_3d

        X2, y2, l2, _, D2, poly2 = generate_spt_dataset(
            n_samples_per_class=n_2d,
            min_length=50,
            max_length=self.max_length,
            dimensionality='2D',
            polymerization_degree=polymerization_degree,
            dt=0.01,
            localization_precision=0.015,
            boost_classes=None,
            verbose=verbose,
            use_full_D_range=True,
            augment_polymerization=True
        )
        X3, y3, l3, _, D3, poly3 = generate_spt_dataset(
            n_samples_per_class=n_3d,
            min_length=50,
            max_length=self.max_length,
            dimensionality='3D',
            polymerization_degree=polymerization_degree,
            dt=0.01,
            localization_precision=0.015,
            boost_classes=None,
            verbose=verbose,
            use_full_D_range=True,
            augment_polymerization=True
        )

        X_traj_list = X2 + X3
        y = np.concatenate([y2, y3])
        lengths = np.concatenate([l2, l3])
        self.class_names = ['Normal', 'Subdiffusion', 'Superdiffusion', 'Confined']

        idx = np.arange(len(y))
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(idx)
        X_traj_list = [X_traj_list[i] for i in idx]
        y = y[idx]
        lengths = lengths[idx]

        if verbose:
            print(f"-> {len(X_traj_list)} Trajektorien generiert")
            print(f"   LÃ¤ngen: {int(lengths.min())}-{int(lengths.max())} Frames")

        if verbose:
            print("\n" + "="*80)
            print("FEATURE-EXTRAKTION")
            print("="*80)
        X_feat = self.feature_extractor.extract_batch(X_traj_list)
        is_3d = np.array([1 if traj.shape[1] == 3 else 0 for traj in X_traj_list], dtype=float).reshape(-1, 1)
        X_feat = np.concatenate([X_feat, is_3d], axis=1)
        self.n_features = X_feat.shape[1]

        n_samples = len(X_traj_list)
        X_traj_padded = np.zeros((n_samples, self.max_length, 3), dtype=np.float32)
        for i, traj in enumerate(X_traj_list):
            if traj.shape[1] == 2:
                traj3 = np.zeros((len(traj), 3), dtype=traj.dtype)
                traj3[:, :2] = traj
            else:
                traj3 = traj
            L = len(traj3)
            if L > self.max_length:
                start = np.random.randint(0, L - self.max_length + 1)
                segment = traj3[start:start + self.max_length]
                X_traj_padded[i, :, :] = segment.astype(np.float32)
            else:
                length = L
                X_traj_padded[i, :length, :] = traj3[:length].astype(np.float32)

        X_traj_temp, X_traj_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
            X_traj_padded, X_feat, y, test_size=0.15, stratify=y, random_state=self.random_seed
        )
        X_traj_train, X_traj_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
            X_traj_temp, X_feat_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=self.random_seed
        )

        X_feat_train = self.scaler.fit_transform(X_feat_train)
        X_feat_val = self.scaler.transform(X_feat_val)
        X_feat_test = self.scaler.transform(X_feat_test)

        self.X_traj_train = X_traj_train
        self.X_feat_train = X_feat_train
        self.y_train = y_train
        self.X_traj_val = X_traj_val
        self.X_feat_val = X_feat_val
        self.y_val = y_val
        self.X_traj_test = X_traj_test
        self.X_feat_test = X_feat_test
        self.y_test = y_test

        if verbose:
            print("\n-> Daten erfolgreich vorbereitet!")

    def augment_training_with_class(self, target_class_index: int, n_new: int = 1000, ratio_3d: float = 0.5, polymerization_degree: float = 0.5, verbose: bool = True):
        """Generiert neue Trajektorien für die schwächste Klasse und fügt sie dem Train-Set hinzu.
        - target_class_index: 0=Normal,1=Subdiffusion,2=Superdiffusion,3=Confined
        - n_new: Anzahl neuer Trainingsbeispiele für diese Klasse (gesamt über 2D+3D)
        - ratio_3d: Anteil 3D (0..1). Rest wird als 2D generiert.
        """
        n3d = int(round(n_new * ratio_3d))
        n2d = n_new - n3d

        def sample_class(dimensionality: str, need: int):
            X_acc, y_acc = [], []
            lengths_acc = []
            # Generator produziert alle Klassen; wir filtern gezielt auf target_class_index
            # Wir rufen iterativ mit moderater Stückzahl auf, bis genug gesammelt.
            chunk = max(50, need)  # pro Klasse
            D_acc = []
            poly_acc = []
            while len(X_acc) < need:
                X_list, y_arr, lengths_arr, _, D_arr, poly_arr = generate_spt_dataset(
                    n_samples_per_class=chunk,
                    min_length=50,
                    max_length=self.max_length,
                    dimensionality=dimensionality,
                    polymerization_degree=polymerization_degree,
                    dt=0.01,
                    localization_precision=0.015,
                    boost_classes=None,
                    verbose=False,
                    use_full_D_range=True,
                    augment_polymerization=True
                )
                for X_i, y_i, L_i, D_i, poly_i in zip(X_list, y_arr, lengths_arr, D_arr, poly_arr):
                    if y_i == target_class_index:
                        X_acc.append(X_i)
                        y_acc.append(y_i)
                        lengths_acc.append(L_i)
                        D_acc.append(D_i)
                        poly_acc.append(poly_i)
                        if len(X_acc) >= need:
                            break
            return X_acc, np.array(y_acc), np.array(lengths_acc), np.array(D_acc), np.array(poly_acc)

        X2_list, y2, _, D2, poly2 = [], np.array([]), None, np.array([]), np.array([])
        X3_list, y3, _, D3, poly3 = [], np.array([]), None, np.array([]), np.array([])
        if n2d > 0:
            X2_list, y2, _, D2, poly2 = sample_class('2D', n2d)
        if n3d > 0:
            X3_list, y3, _, D3, poly3 = sample_class('3D', n3d)

        X_new_list = X2_list + X3_list
        y_new = np.concatenate([y2, y3]) if y2.size or y3.size else np.array([])
        D_new = np.concatenate([D2, D3]) if D2.size or D3.size else np.array([])
        poly_new = np.concatenate([poly2, poly3]) if poly2.size or poly3.size else np.array([])

        # Shuffle neu generierte
        if len(X_new_list) == 0:
            if verbose:
                print("Keine neuen Beispiele generiert.")
            return
        idx = np.arange(len(X_new_list))
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(idx)
        X_new_list = [X_new_list[i] for i in idx]
        y_new = y_new[idx]
        D_new = D_new[idx]
        poly_new = poly_new[idx]

        # Feature-Extraktion + D, Poly, Dim Features
        X_new_feat = self.feature_extractor.extract_batch(X_new_list)
        D_log = np.log10(D_new).reshape(-1, 1)
        poly_feat = poly_new.reshape(-1, 1)
        dim_feat = np.array([0.0 if traj.shape[1] == 2 else 1.0 for traj in X_new_list]).reshape(-1, 1)
        X_new_feat = np.concatenate([X_new_feat, D_log, poly_feat, dim_feat], axis=1)
        # Transform mit bestehendem Scaler (nicht neu fitten, um Val/Test konsistent zu halten)
        X_new_feat_scaled = self.scaler.transform(X_new_feat)

        # Padding Trajektorien (immer 3 Kanäle)
        X_new_traj = np.zeros((len(X_new_list), self.max_length, 3), dtype=np.float32)
        for i, traj in enumerate(X_new_list):
            if traj.shape[1] == 2:
                traj3 = np.zeros((len(traj), 3), dtype=traj.dtype)
                traj3[:, :2] = traj
            else:
                traj3 = traj
            L = len(traj3)
            if L > self.max_length:
                start = np.random.randint(0, L - self.max_length + 1)
                segment = traj3[start:start + self.max_length]
                X_new_traj[i, :, :] = segment.astype(np.float32)
            else:
                X_new_traj[i, :L, :] = traj3[:L].astype(np.float32)

        # An Train-Set anhängen und durchmischen
        self.X_traj_train = np.concatenate([self.X_traj_train, X_new_traj], axis=0)
        self.X_feat_train = np.concatenate([self.X_feat_train, X_new_feat_scaled], axis=0)
        self.y_train = np.concatenate([self.y_train, y_new], axis=0)

        perm = np.random.permutation(len(self.y_train))
        self.X_traj_train = self.X_traj_train[perm]
        self.X_feat_train = self.X_feat_train[perm]
        self.y_train = self.y_train[perm]

        if verbose:
            cname = self.class_names[target_class_index] if target_class_index < len(self.class_names) else str(target_class_index)
            print(f"? Augment: +{len(X_new_list)} Samples für schwächste Klasse '{cname}' (2D/3D={n2d}/{n3d})")
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        epoch_callback=None
    ):
        """
        Trainiere Modell mit Best Practices
        
        Args:
            epochs: Maximale Anzahl Epochen
            batch_size: Batch Size
            verbose: Keras Verbosity (0, 1, 2)
            epoch_callback: Optionaler Callback ``fn(epoch, logs, total_epochs)``
                für externe Fortschrittsvisualisierung (z.B. GUI).
            
        Training-Strategie:
        -------------------
        
        1. **Early Stopping**: Stoppt bei Validierungs-Loss Plateau
           Patience: 15 Epochen
           Restore Best Weights: Ja
           
        2. **Learning Rate Reduction**: Bei Validierungs-Loss Plateau
           Factor: 0.5 (halbiert LR)
           Patience: 8 Epochen
           Min LR: 1e-7
           
        3. **Model Checkpoint**: Speichert bestes Modell
           Monitor: val_accuracy
           Mode: max
           
        4. **Class Weighting**: Korrigiert fÃ¼r Klassenimbalance
           Automatisch berechnet aus Training-Labels
        
        Theoretische Rechtfertigung:
        ----------------------------
        - **Early Stopping**: Verhindert Overfitting (Prechelt, 1998)
        - **LR Scheduling**: Verbessert Konvergenz (Smith, 2017)
        - **Batch Normalization**: Reduziert Internal Covariate Shift (Ioffe & Szegedy, 2015)
        - **Dropout**: Ensemble-Effekt, verhindert Co-Adaptation (Srivastava et al., 2014)
        """
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Optimizer: Adam (lr=1e-3)")
        print("Callbacks: Early Stopping, LR Reduction, Checkpointing")
        print("="*80 + "\n")
        
        # Compute Class Weights
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        
        print("Class Weights (fÃ¼r Imbalance-Korrektur):")
        for i, name in enumerate(self.class_names):
            print(f"  {name}: {class_weights[i]:.3f}")
        print()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Training
        callback_list = list(callbacks)

        if epoch_callback is not None:
            class EpochBridgeCallback(keras.callbacks.Callback):
                def __init__(self, total_epochs: int, cb):
                    super().__init__()
                    self.total_epochs = total_epochs
                    self.cb = cb

                def on_epoch_end(self, epoch, logs=None):
                    try:
                        self.cb(epoch, logs or {}, self.total_epochs)
                    except Exception:
                        # Callback-Fehler sollen das Training nicht abbrechen
                        pass

            callback_list.append(EpochBridgeCallback(epochs, epoch_callback))

        history = self.model.fit(
            [self.X_traj_train, self.X_feat_train],
            self.y_train,
            validation_data=([self.X_traj_val, self.X_feat_val], self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.history = history
        
        print("\nâœ… Training abgeschlossen!")
        
        # Plot Training History
        self.plot_training_history()
    
    def plot_training_history(self):
        """Visualisiere Training History"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Acc', linewidth=2)
        ax2.plot(self.history.history['val_accuracy'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=150)
        print(f"âœ… Training History Plot gespeichert: {self.output_dir}/training_history.png")
        try:
            plt.close()
        except Exception:
            pass


    
    def evaluate(self, verbose: bool = True):
        """
        Evaluiere Modell auf Test-Set
        
        Metriken:
        ---------
        1. **Overall Accuracy**: Gesamtgenauigkeit
        2. **Per-Class Metrics**: Precision, Recall, F1-Score pro Klasse
        3. **Confusion Matrix**: Fehleranalyse
        
        Wissenschaftliche Interpretation:
        ---------------------------------
        - **Precision**: P(true class | predicted class) - Wichtig fÃ¼r Anwendungen
          wo False Positives kritisch sind
        - **Recall**: P(predicted class | true class) - Wichtig fÃ¼r vollstÃ¤ndige Detektion
        - **F1-Score**: Harmonisches Mittel von Precision und Recall
          F1 = 2Â·(PrecisionÂ·Recall)/(Precision+Recall)
        """
        if verbose:
            print("\n" + "="*80)
            print("EVALUATION AUF TEST-SET")
            print("="*80)
        
        # Predictions
        y_pred_probs = self.model.predict(
            [self.X_traj_test, self.X_feat_test],
            verbose=0
        )
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Overall Accuracy
        accuracy = np.mean(y_pred == self.y_test)
        
        if verbose:
            print(f"\nðŸ“Š OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification Report
        if verbose:
            print("\n" + "="*80)
            print("CLASSIFICATION REPORT")
            print("="*80)
        
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        if verbose:
            print(report)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        try:
            import seaborn as sns
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar_kws={'label': 'Count'}
            )
        except Exception:
            plt.imshow(cm, cmap='Blues')
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
            plt.xticks(np.arange(len(self.class_names)), self.class_names)
            plt.yticks(np.arange(len(self.class_names)), self.class_names)
            plt.colorbar(label='Count')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150)
        print(f"\nâœ… Confusion Matrix gespeichert: {self.output_dir}/confusion_matrix.png")
        try:
            plt.close()
        except Exception:
            pass


        
        return accuracy, report, cm
    
    def save_model(self, verbose: bool = True):
        """
        Speichere trainiertes Modell und Metadaten
        
        Gespeichert werden:
        -------------------
        1. **Model (.keras)**: VollstÃ¤ndiges Keras-Modell
        2. **Scaler (.pkl)**: StandardScaler fÃ¼r Features
        3. **Feature Names (.pkl)**: Namen der 24 Features
        4. **Metadata (.json)**: Konfiguration, Datum, Performance
        5. **Training History (.pkl)**: Loss/Accuracy VerlÃ¤ufe
        """
        if verbose:
            print("\n" + "="*80)
            print("MODELL SPEICHERN")
            print("="*80)
        
        # Model
        model_path = os.path.join(self.output_dir, 'spt_classifier.keras')
        self.model.save(model_path)
        if verbose:
            print(f"âœ… Modell gespeichert: {model_path}")
        
        # Scaler
        scaler_path = os.path.join(self.output_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        if verbose:
            print(f"âœ… Feature Scaler gespeichert: {scaler_path}")
        
        # Feature Names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.pkl')
        with open(feature_names_path, 'wb') as f:
            pickle.dump(list(self.feature_extractor.feature_names) + ['is_3D'], f)
        if verbose:
            print(f"âœ… Feature Names gespeichert: {feature_names_path}")
        
        # Metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'max_length': self.max_length,
            'n_features': self.n_features,
            'input_dim': self.input_dim,
            'class_names': self.class_names,
            'random_seed': self.random_seed,
            'total_parameters': self.model.count_params()
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"âœ… Metadata gespeichert: {metadata_path}")
        
        # Training History
        if self.history is not None:
            history_path = os.path.join(self.output_dir, 'training_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            if verbose:
                print(f"âœ… Training History gespeichert: {history_path}")
        
        if verbose:
            print("="*80)
            print(f"âœ… ALLE DATEIEN IN: {self.output_dir}/")
            print("="*80)


def run_complete_training(
    n_samples_per_class: int = 3000,
    dimensionality: str = '2D',
    polymerization_degree: float = 0.5,
    epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = './spt_trained_model'
):
    """
    FÃ¼hre komplettes End-to-End Training aus
    
    Args:
        n_samples_per_class: Samples pro Diffusionsklasse
        dimensionality: '2D' oder '3D'
        polymerization_degree: 0.0-1.0 (Polymer-Zustand)
        epochs: Max Epochen
        batch_size: Batch Size
        output_dir: Output-Verzeichnis
        
    Returns:
        trainer: SPTClassifierTrainer Instanz mit trainiertem Modell
    """
    print("\n" + "="*80)
    print("SPT CLASSIFIER - KOMPLETTES TRAINING")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialisiere Trainer
    trainer = SPTClassifierTrainer(
        max_length=2000,
        output_dir=output_dir,
        random_seed=42
    )
    
    # 1. Datengenerierung
    trainer.generate_training_data(
        n_samples_per_class=n_samples_per_class,
        mode=dimensionality,
        ratio_3d=1.0 if dimensionality.lower() == '3d' else 0.0,
        polymerization_degree=polymerization_degree,
        verbose=True
    )
    
    # 2. Modell-Konstruktion
    trainer.build_model(verbose=True)
    
    # 3. Training
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # 4. Evaluation
    trainer.evaluate(verbose=True)
    
    # 5. Speichern
    trainer.save_model(verbose=True)
    
    print("\n" + "="*80)
    print("âœ… TRAINING ERFOLGREICH ABGESCHLOSSEN!")
    print(f"Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return trainer


def run_complete_training_both(
    n_samples_per_class: int = 10000,
    ratio_3d: float = 0.5,
    polymerization_degree: float = 0.5,
    epochs_per_round: int = 200,
    max_rounds: int = 15,
    target_overall: float = 0.97,
    target_min_f1: float = 0.95,
    batch_size: int = 512,
    output_dir: str = './spt_trained_model'
):
    """End-to-end Training fÃ¼r gemischte 2D+3D Daten mit iterativer Optimierung."""
    print("\n" + "="*80)
    print("SPT CLASSIFIER - KOMPLETTES TRAINING (2D+3D)")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    trainer = SPTClassifierTrainer(
        max_length=2000,
        output_dir=output_dir,
        random_seed=42
    )

    trainer.generate_training_data_both(
        n_samples_per_class=n_samples_per_class,
        ratio_3d=ratio_3d,
        polymerization_degree=polymerization_degree,
        verbose=True
    )

    trainer.build_model(verbose=True)

    # Iteratives Training gegen Validierungs-Set
    for r in range(1, max_rounds + 1):
        print("\n" + "-"*80)
        print(f"RUNDE {r}/{max_rounds}")
        print("-"*80)
        trainer.train(epochs=epochs_per_round, batch_size=batch_size, verbose=1)

        # Val-Auswertung
        y_val_pred = np.argmax(trainer.model.predict([trainer.X_traj_val, trainer.X_feat_val], verbose=0), axis=1)
        acc_val = np.mean(y_val_pred == trainer.y_val)
        rep = classification_report(trainer.y_val, y_val_pred, target_names=trainer.class_names, output_dict=True)
        min_f1 = min(rep[c]['f1-score'] for c in trainer.class_names)
        print(f"â†’ Val Accuracy: {acc_val:.4f}; Min per-class F1: {min_f1:.4f}")
        if acc_val >= target_overall and min_f1 >= target_min_f1:
            print("Ziele erreicht â€“ beende iteratives Training.")
            break
        else:
            print("Ziele noch nicht erreicht â€“ weitere Runde.")
            # adaptiv neue Daten für schwächste Klasse (geringster F1) generieren
            rep = classification_report(trainer.y_val, y_val_pred, target_names=trainer.class_names, output_dict=True)
            f1s = [rep[c]['f1-score'] for c in trainer.class_names]
            weakest_idx = int(np.argmin(f1s))
            trainer.augment_training_with_class(
                target_class_index=weakest_idx,
                n_new=[int(0.2*len(trainer.y_train)), 500][1] if len(trainer.y_train) < 2500 else int(0.2*len(trainer.y_train)),
                ratio_3d=ratio_3d,
                polymerization_degree=polymerization_degree,
                verbose=True
            )

    # Test-Evaluation und Speichern
    trainer.evaluate(verbose=True)
    trainer.save_model(verbose=True)

    print("\n" + "="*80)
    print("âœ“ TRAINING ERFOLGREICH ABGESCHLOSSEN!")
    print(f"Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return trainer


# =============================================================================
# MAIN: AUSFÃœHRUNG
# =============================================================================

if __name__ == '__main__':
    # Ressourcenschonende Defaults, gemischtes 2D+3D Training
    CONFIG = {
        'n_samples_per_class': 10000,
        'ratio_3d': 0.5,
        'polymerization_degree': 0.5,
        'epochs_per_round': 200,
        'max_rounds': 15,
        'target_overall': 0.97,
        'target_min_f1': 0.95,
        'batch_size': 512,
        'output_dir': './spt_trained_model'
    }

    print("\nKONFIGURATION:")
    for key, val in CONFIG.items():
        print(f"  {key}: {val}")

    trainer = run_complete_training_both(**CONFIG)

    print("\n" + "="*80)
    print("AUSGABE-DATEIEN:")
    print("="*80)
    print(f"  • spt_classifier.keras         - Trainiertes Modell")
    print(f"  • feature_scaler.pkl           - Feature Standardisierung")
    print(f"  • feature_names.pkl            - Feature-Namen Liste")
    print(f"  • metadata.json                - Modell-Metadaten")
    print(f"  • training_history.pkl         - Training Verlauf")
    print(f"  • training_history.png         - Loss/Accuracy Plots")
    print(f"  • confusion_matrix.png         - Confusion Matrix")
    print("="*80)
    print(f"\nAlle Dateien in: {CONFIG['output_dir']}/")
    print("="*80)



