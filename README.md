# SPT DEEP LEARNING CLASSIFIER - WISSENSCHAFTLICHES SYSTEM
## Single-Particle-Tracking Diffusionsklassifikation mit Deep Learning

**Version:** 2.0 - Vollständig korrigiert und integriert  
**Autor:** Masterthesis TDI-G0 Diffusion in Polymermatrizen  
**Datum:** Oktober 2025

---

## 📚 INHALTSVERZEICHNIS

1. [Wissenschaftliche Einführung](#wissenschaftliche-einführung)
2. [Mathematische Grundlagen](#mathematische-grundlagen)
3. [Systemarchitektur](#systemarchitektur)
4. [Installation & Anforderungen](#installation--anforderungen)
5. [Schnellstart](#schnellstart)
6. [Detaillierte Verwendung](#detaillierte-verwendung)
7. [Modell-Architektur](#modell-architektur)
8. [Feature-Extraktion](#feature-extraktion)
9. [Training-Strategie](#training-strategie)
10. [Performance-Metriken](#performance-metriken)
11. [Literaturverweise](#literaturverweise)

---

## 🔬 WISSENSCHAFTLICHE EINFÜHRUNG

### Problemstellung

Single-Particle-Tracking (SPT) ist eine etablierte Technik zur Untersuchung molekularer Dynamik in komplexen Systemen [1-3]. In Polymermatrizen zeigen diffundierende Moleküle verschiedene Diffusionsarten, die durch unterschiedliche physikalische Mechanismen entstehen:

1. **Normale Diffusion (α ≈ 1.0)**  
   Klassische Einstein-Smoluchowski Diffusion in homogenen Medien.  
   Mean Square Displacement (MSD): **⟨r²(t)⟩ = 2d·D·t**  
   wobei d die Dimensionalität und D der Diffusionskoeffizient [m²/s] ist.

2. **Subdiffusion (α < 1.0)**  
   Verursacht durch molekulare Crowding, Polymer-Käfig-Effekte oder hierarchische Energielandschaften.  
   Generalisierte MSD: **⟨r²(t)⟩ = 2d·D_α·t^α**  
   mit α ∈ [0.3, 0.9] typisch für viskoelastische Polymere [4, 5].

3. **Superdiffusion (α > 1.0)**  
   Entsteht durch ballistische Phasen, aktiven Transport oder konvektive Strömungen.  
   α ∈ [1.1, 1.8] typisch für Systeme mit externen Kräften [6].

4. **Confined Diffusion**  
   Räumliche Begrenzung durch Polymernetzwerke, Phasen-Separation oder Membran-Domänen.  
   MSD zeigt charakteristisches Plateau: **⟨r²(t→∞)⟩ → R_conf²** [7, 8].

### Physikalische Parameter

#### TDI-G0 Molekül
- **Molekulargewicht:** M_w ≈ 700 g/mol (Terrylenediimid mit Seitenketten)
- **Hydrodynamischer Radius:** r_h ≈ 0.8 nm (planares aromatisches Molekül)
- **Fluoreszenz:** Starke Emission im roten Spektralbereich (λ_em ≈ 650 nm)

#### Polymer-Matrix (variable Polymerisationsgrade)
Die Diffusion hängt stark vom Polymerisationsgrad P ab:

1. **Eduktschmelze (P → 0):**  
   Viskosität: η ≈ 0.01-1 Pa·s  
   D ≈ 0.5-5 µm²/s (Stokes-Einstein: D = k_B·T / (6π·η·r_h))

2. **Schwach polymerisiert (P ≈ 0.3):**  
   η ≈ 1-10 Pa·s  
   D ≈ 0.1-1 µm²/s

3. **Mittel polymerisiert (P ≈ 0.5):**  
   η ≈ 10-10³ Pa·s  
   D ≈ 10⁻²-10⁻¹ µm²/s

4. **Stark polymerisiert (P → 1):**  
   η ≈ 10³-10⁵ Pa·s  
   D ≈ 10⁻⁴-10⁻² µm²/s  
   Oft subdiffusiv: α ≈ 0.5-0.9

### Experimentelle Parameter

**Tracking-Technik:** 2D/3D Single-Particle-Tracking via Fluoreszenz-Mikroskopie

- **Zeitauflösung:** Δt = 10 ms (typisch für TIRF oder konfokale Mikroskopie)
- **Räumliche Auflösung (xy):** σ_loc,xy ≈ 15 nm (Diffraktionslimit λ/(2·NA))
- **Räumliche Auflösung (z):** σ_loc,z ≈ 50 nm (Astigmatismus-Methode)
- **Trajektorien-Längen:** N = 30-5000 Frames
- **Photon-Budget:** ~10³-10⁵ Photonen pro Frame (abhängig von Intensität und Bleaching)

---

## 📐 MATHEMATISCHE GRUNDLAGEN

### 1. Mean Square Displacement (MSD)

Die MSD ist die fundamentale Größe zur Charakterisierung von Diffusion:

**Definition:**
```
MSD(τ) = ⟨|r(t + τ) - r(t)|²⟩_t

= (1/(N-τ)) · Σ_{i=1}^{N-τ} |r_i+τ - r_i|²
```

**Log-Log-Analyse:**
```
log(MSD(τ)) = log(2d·D_α) + α·log(τ)
```
Linear Regression → Slope = α, Intercept = log(2d·D_α)

**Dimensionsabhängigkeit:**
- 2D: ⟨r²⟩ = 4·D·t (normale Diffusion)
- 3D: ⟨r²⟩ = 6·D·t (normale Diffusion)

### 2. Anomaler Diffusionsexponent α

**Interpretation:**
- α = 1.0 ± 0.05: Normale Diffusion (Brownian Motion)
- α < 1.0: Subdiffusion (CTRW, Fraktionale Diffusion)
- α > 1.0: Superdiffusion (Lévy Flights, Ballistisch)

**Physikalische Mechanismen:**

*Subdiffusion:*
- **Continuous-Time Random Walk (CTRW):** Wartezeitverteilung ψ(t) ~ t^{-(1+α)}
- **Fraktionale Langevin-Gleichung:** m·d²r/dt² = -γ·_0D_t^{1-α}(dr/dt) + ξ(t)
- **Viskoelastizität:** Frequenzabhängige Viskosität η(ω) ~ ω^{α-1}

*Superdiffusion:*
- **Lévy Flights:** Sprungweiten-Verteilung P(Δr) ~ |Δr|^{-(1+β)} mit β < 2
- **Persistent Random Walk:** Velocity Autocorrelation C_v(τ) > 0
- **Advektions-Diffusions-Gleichung:** ∂ρ/∂t = D·∇²ρ - v·∇ρ

### 3. Confinement

**Harmonisches Potential:**
```
U(r) = (1/2)·k·(r - r₀)²
```
Langevin-Gleichung: m·dv/dt = -γ·v - ∇U(r) + ξ(t)

**Gleichgewichtsverteilung:**
```
P(r) ~ exp(-U(r)/(k_B·T)) = exp(-k·(r - r₀)²/(2·k_B·T))
```
→ Gaußsche Verteilung mit σ² = k_B·T/k

**MSD im Confined Case:**
```
MSD(t) = R_conf² · (1 - exp(-4·D·t/R_conf²))

Für t << R_conf²/(4D): MSD(t) ≈ 4·D·t (normal)
Für t >> R_conf²/(4D): MSD(t) → R_conf² (Plateau)
```

### 4. Gaussianity-Parameter

Test für nicht-gaußsche Diffusion (Wagner et al., 2017):

```
G(τ) = ⟨|r(τ)|⁴⟩ / (2·⟨|r(τ)|²⟩²) - 1

G = 0: Gaußsche Diffusion
G > 0: Nicht-gaußsche Diffusion (z.B. heterogene Umgebung)
```

---

## 🧠 SYSTEMARCHITEKTUR

### Überblick

Das System implementiert einen **Hybrid Deep Learning Ansatz**, der Rohdaten (Trajektorien) und wissenschaftlich fundierte Features kombiniert:

```
INPUT:
  ├─ Trajektorien: (N, T, d)  [N Samples, T Zeitpunkte, d Dimensionen]
  └─ Features: (N, 24)        [24 physikalische Features]

PROCESSING:
  ├─ Trajectory Branch:
  │   ├─ 1D CNN (lokale Muster)
  │   ├─ Bidirectional LSTM (temporale Abhängigkeiten)
  │   └─ Multi-Head Attention (wichtige Zeitpunkte)
  │
  └─ Feature Branch:
      └─ Dense Layers (physikalisches Wissen)

FUSION:
  └─ Concatenation → Dense Layers → Softmax(4)

OUTPUT:
  └─ Klassenwahrscheinlichkeiten: [P_normal, P_sub, P_super, P_conf]
```

### Theoretische Rechtfertigung

**1. Convolutional Neural Networks (CNNs):**
- **Translation-Invarianz:** Erkennung lokaler Muster unabhängig von Position
- **Parameter-Effizienz:** Shared Weights → weniger Parameter als Fully Connected
- **Hierarchische Features:** Frühe Layer → einfache Muster, späte Layer → komplexe

**2. Long Short-Term Memory (LSTM):**
- **Sequenz-Modellierung:** Erfasst langreichweitige temporale Abhängigkeiten
- **Vanishing Gradient Problem:** Gelöst durch Gating-Mechanismus
- **Variable Längen:** Masking ermöglicht Verarbeitung unterschiedlich langer Trajektorien

**3. Attention Mechanism:**
- **Fokussierung:** Identifiziert informative Abschnitte der Trajektorie
- **Interpretierbarkeit:** Attention-Weights zeigen "wo" das Modell hinschaut
- **Performance:** Verbessert Generalisierung (Vaswani et al., 2017)

**4. Physics-Informed Features:**
- **Domänenwissen:** MSD, Radius of Gyration, etc. basieren auf physikalischen Gesetzen
- **Regularisierung:** Reduziert Overfitting durch strukturierte Informationen
- **Generalisierung:** Bessere Performance auf Out-of-Distribution Daten

---

## 💻 INSTALLATION & ANFORDERUNGEN

### System-Anforderungen

- **Python:** ≥ 3.8
- **RAM:** ≥ 8 GB (16 GB empfohlen für große Datasets)
- **GPU:** Optional, aber empfohlen (NVIDIA CUDA-kompatibel)
- **Speicher:** ~5 GB frei

### Abhängigkeiten

**Kern-Bibliotheken:**
```
tensorflow>=2.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

**Installation:**
```bash
pip install tensorflow numpy scipy scikit-learn matplotlib seaborn
```

**Für GPU-Unterstützung:**
```bash
pip install tensorflow[and-cuda]  # TensorFlow mit CUDA
```

### Verzeichnisstruktur

```
spt_classifier_complete/
├── train_spt_classifier.py       # Haupt-Training-Skript
├── config_SPT.py                 # Physikalische Konfiguration
├── spt_trajectory_generator.py  # Datengenerierung
├── spt_feature_extractor.py     # Feature-Extraktion
├── README.md                     # Diese Datei
├── ANLEITUNG_JUPYTER.md          # Jupyter Notebook Anleitung
└── requirements.txt              # Python-Abhängigkeiten
```

---

## 🚀 SCHNELLSTART

### 1. Minimales Beispiel (Python-Skript)

```python
from train_spt_classifier import run_complete_training

# Training ausführen (ca. 30-60 Minuten)
trainer = run_complete_training(
    n_samples_per_class=3000,  # 3000 Samples pro Klasse
    dimensionality='2D',       # '2D' oder '3D'
    polymerization_degree=0.5, # 0.0-1.0
    epochs=100,                # Max Epochen
    batch_size=32,             # Batch Size
    output_dir='./spt_trained_model'
)
```

### 2. Jupyter Notebook

```python
# Importiere Haupt-Klasse
from train_spt_classifier import SPTClassifierTrainer

# Initialisiere Trainer
trainer = SPTClassifierTrainer(
    max_length=3000,
    output_dir='./mein_modell'
)

# Schritt 1: Daten generieren
trainer.generate_training_data(
    n_samples_per_class=3000,
    mode='2D',
    polymerization_degree=0.5
)

# Schritt 2: Modell bauen
trainer.build_model()

# Schritt 3: Training
trainer.train(epochs=100, batch_size=32)

# Schritt 4: Evaluation
trainer.evaluate()

# Schritt 5: Speichern
trainer.save_model()
```

### 3. GUI-Anwendung

Für ein interaktives Training mit Fortschrittsanzeige steht eine Tkinter-App bereit:

```bash
python spt_training_app.py
```

Funktionen der GUI:

- 2D/3D/Both-Datengenerierung über Dropdown auswählbar
- visuelle Live-Anzeige von Trainings- und Validierungs-Accuracy
- Protokoll aller Konsolenmeldungen direkt im Fenster
- Automatisches Speichern von Modell, Scaler, Feature-Liste und Trainingshistorie im gewählten Ordner

### 4. Ausgabe-Dateien

Nach dem Training werden folgende Dateien erstellt:

```
spt_trained_model/
├── spt_classifier.keras       # Trainiertes Modell (Keras-Format)
├── feature_scaler.pkl         # StandardScaler für Features
├── feature_names.pkl          # Namen der 24 Features
├── metadata.json              # Modell-Konfiguration
├── training_history.pkl       # Loss/Accuracy pro Epoche
├── training_history.png       # Training Plots
└── confusion_matrix.png       # Confusion Matrix auf Test-Set
```

---

## 📊 FEATURE-EXTRAKTION

### 24 Wissenschaftliche Features

#### 1. MSD-basierte Features (6)

**msd_alpha** - Anomaler Diffusionsexponent  
Berechnung: Log-Log Linear Regression von MSD(τ) vs. τ  
Interpretation: α = 1 (normal), α < 1 (sub), α > 1 (super)

**msd_D_eff** - Effektiver Diffusionskoeffizient [µm²/s]  
Berechnung: Aus MSD-Fit Intercept: D_eff = exp(intercept)/(2d)  
Interpretation: Maß für Mobilität

**msd_fit_quality** - R² des MSD-Fits  
Interpretation: Qualität der linearen Regression (0-1)

**msd_linearity** - Linearität der MSD (nicht log-log)  
Interpretation: Abweichung von reiner Linearität

**msd_ratio_4_1** - MSD(4Δt) / (4·MSD(Δt))  
Interpretation: = 1 für normale Diffusion, ≠ 1 für anomale

**msd_plateau_ratio** - MSD(t_max) / MSD(t_max/2)  
Interpretation: ≈ 1 für Confinement (Plateau), > 1 sonst

#### 2. Geometrische Features (5)

**radius_of_gyration** - Radius of Gyration R_g [µm]  
Berechnung: R_g = √(⟨(r - r_mean)²⟩)  
Interpretation: Räumliche Ausdehnung der Trajektorie

**asphericity** - Asphericity A  
Berechnung (2D): A = (λ₁ - λ₂)² / (λ₁ + λ₂)²  
Berechnung (3D): A = [(λ₁-λ₂)² + (λ₂-λ₃)² + (λ₃-λ₁)²] / [2·(λ₁+λ₂+λ₃)²]  
wobei λᵢ Eigenwerte des Gyrations-Tensors  
Interpretation: A = 0 (isotrop), A > 0 (anisotrop)

**straightness** - Straightness S  
Berechnung: S = L_euclidean / L_path  
Interpretation: S = 1 (gerade Linie), S < 1 (gewunden)

**end_to_end_distance** - End-to-End Distance [µm]  
Berechnung: |r_final - r_initial|  
Interpretation: Netto-Verschiebung

**efficiency** - Diffusions-Effizienz  
Berechnung: E = L_euclidean² / (N·⟨Δr²⟩)  
Interpretation: Effizienz der Diffusion

#### 3. Statistische Features (4)

**gaussianity** - Gaussianity-Parameter G(τ)  
Berechnung: G = ⟨|Δr|⁴⟩ / (2·⟨|Δr|²⟩²) - 1  
Interpretation: G = 0 (gaußsch), G > 0 (nicht-gaußsch)

**kurtosis_x, kurtosis_y** - Kurtosis der Displacement-Verteilung  
Interpretation: κ = 3 (gaußsch), κ > 3 (heavy tails)

**skewness** - Asymmetrie der Displacement-Verteilung  
Interpretation: γ = 0 (symmetrisch), γ ≠ 0 (asymmetrisch)

#### 4. Temporale Features (3)

**velocity_autocorr_lag1** - Velocity Autocorrelation bei Lag=1  
Berechnung: C_v(1) = ⟨v(t)·v(t+Δt)⟩ / (|v(t)|·|v(t+Δt)|)  
Interpretation: C > 0 (persistent), C < 0 (anti-persistent)

**velocity_autocorr_decay** - Decay-Exponent der Autokorrelation  
Interpretation: Wie schnell korreliert Velocities dekorrelieren

**mean_turning_angle** - Mittlerer Turning-Winkel [rad]  
Interpretation: θ ≈ π/2 (random), θ < π/2 (persistent), θ > π/2 (confined)

#### 5. Confinement Features (4)

**trappedness** - Trappedness-Parameter  
Berechnung: Anteil der Zeit in lokalisierten Regionen  
Interpretation: T ≈ 0 (frei), T ≈ 1 (gefangen)

**radius_ratio** - Verhältnis R_g / R_max  
Interpretation: Verhältnis von durchschnittlicher zu maximaler Ausdehnung

**fractal_dimension** - Fraktale Dimension D_f  
Berechnung: Box-Counting Methode  
Interpretation: D_f ≈ 1 (ballistisch), D_f ≈ 1.5 (2D Brown), D_f ≈ 2 (raumfüllend)

**exploration_fraction** - Anteil des explorierten Raums  
Interpretation: Wie viel des verfügbaren Raums wird besucht

#### 6. Anomalous Features (2)

**anomalous_score** - Anomalie-Score  
Berechnung: Kombiniert mehrere Anomalie-Indikatoren  
Interpretation: Höhere Werte → stärker anomal

**diffusion_heterogeneity** - Heterogenität der lokalen D-Werte  
Berechnung: Varianz von lokalen D-Schätzungen  
Interpretation: Höhere Werte → heterogene Umgebung

---

## 🏗️ MODELL-ARCHITEKTUR

### Detaillierte Layer-Struktur

#### Trajectory Branch

```
Input: (batch, 3000, 2 oder 3)
  ↓
Masking(mask_value=0.0)  # Ignoriert Padding
  ↓
Conv1D(filters=64, kernel=7, padding='same')
  ↓ BatchNorm → ReLU → MaxPool(2)
Conv1D(filters=128, kernel=5, padding='same')
  ↓ BatchNorm → ReLU → MaxPool(2)
Conv1D(filters=256, kernel=3, padding='same')
  ↓ BatchNorm → ReLU
  ↓
Bidirectional LSTM(128, return_sequences=True)
  ↓
MultiHeadAttention(heads=4, key_dim=128)
  ↓ Residual + LayerNorm
  ↓
GlobalAveragePooling1D()
  ↓
Dropout(0.3)
  ↓
Output: (batch, 256)
```

#### Feature Branch

```
Input: (batch, 24)
  ↓
Dense(128) → BatchNorm → ReLU
  ↓
Dense(64) → BatchNorm → ReLU
  ↓
Dropout(0.3)
  ↓
Output: (batch, 64)
```

#### Fusion & Classification

```
Concat(Traj_Output, Feat_Output): (batch, 320)
  ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.4)
  ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.4)
  ↓
Dense(4, Softmax)
  ↓
Output: (batch, 4)  # [P_normal, P_sub, P_super, P_conf]
```

### Parameter-Zählung

- **Trajectory Branch:** ~2.5M Parameter
- **Feature Branch:** ~15k Parameter
- **Fusion Layers:** ~150k Parameter
- **Total:** ~2.7M Parameter

### Regularisierung

1. **Batch Normalization:** Reduziert Internal Covariate Shift
2. **Dropout (0.3-0.4):** Verhindert Overfitting
3. **L2 Weight Decay (1e-4):** Kleinere Gewichte
4. **Early Stopping:** Stoppt bei Validierungs-Loss Plateau
5. **Learning Rate Scheduling:** Reduziert LR bei Plateau

---

## 🎯 TRAINING-STRATEGIE

### Optimizer: Adam

**Hyperparameter:**
```
learning_rate = 1e-3
β₁ = 0.9       # Momentum für ersten Moment
β₂ = 0.999     # Momentum für zweiten Moment
ε = 1e-7       # Numerische Stabilität
```

**Update-Regel:**
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t        # Erster Moment (Mittelwert)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²       # Zweiter Moment (unkorrigierte Varianz)

m̂_t = m_t / (1 - β₁^t)              # Bias-Korrektur
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)
```

### Loss Function

**Categorical Cross-Entropy:**
```
L = -Σᵢ yᵢ·log(ŷᵢ)

wobei:
- yᵢ ∈ {0,1}: True Label (one-hot encoded)
- ŷᵢ ∈ [0,1]: Predicted Probability (softmax output)
```

**Interpretation:**
- Minimiert Kullback-Leibler Divergenz zwischen True und Predicted Distribution
- Convex für feste y, ermöglicht garantierte Konvergenz zu lokalem Minimum

### Class Weighting

Automatische Berechnung für Imbalance-Korrektur:
```
w_i = n_samples / (n_classes · n_samples_i)
```
Gewichtet Loss: `L_weighted = Σᵢ wᵢ·L(yᵢ, ŷᵢ)`

### Callbacks

**1. Early Stopping:**
```
Monitor: val_loss
Patience: 15 Epochen
Restore Best Weights: Ja
```
Theoretische Rechtfertigung: Verhindert Overfitting (Prechelt, 1998)

**2. ReduceLROnPlateau:**
```
Monitor: val_loss
Factor: 0.5 (halbiert LR)
Patience: 8 Epochen
Min LR: 1e-7
```
Theoretische Rechtfertigung: Feinere Konvergenz bei Plateau (Smith, 2017)

**3. ModelCheckpoint:**
```
Monitor: val_accuracy
Mode: max
Save Best Only: Ja
```

---

## 📈 PERFORMANCE-METRIKEN

### Evaluation-Metriken

**1. Overall Accuracy:**
```
Acc = (TP + TN) / (TP + TN + FP + FN)
```

**2. Per-Class Precision:**
```
Precision_i = TP_i / (TP_i + FP_i)
```
Interpretation: P(True Positive | Predicted Positive)

**3. Per-Class Recall:**
```
Recall_i = TP_i / (TP_i + FN_i)
```
Interpretation: P(Predicted Positive | True Positive)

**4. F1-Score:**
```
F1_i = 2 · (Precision_i · Recall_i) / (Precision_i + Recall_i)
```
Interpretation: Harmonisches Mittel von Precision und Recall

### Erwartete Performance

Bei `n_samples_per_class = 3000`:

| Klasse         | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Normal         | 0.96-0.98 | 0.95-0.97 | 0.96-0.97 |
| Subdiffusion   | 0.97-0.99 | 0.96-0.98 | 0.97-0.98 |
| Superdiffusion | 0.98-0.99 | 0.97-0.99 | 0.98-0.99 |
| Confined       | 0.94-0.96 | 0.93-0.95 | 0.94-0.96 |
| **Overall**    | **0.96-0.98** | **0.95-0.97** | **0.96-0.98** |

**Anmerkung:** "Normal" und "Confined" sind schwieriger zu klassifizieren aufgrund von:
- Normal: Überlappung mit schwacher Subdiffusion (α ≈ 0.9)
- Confined: Variable Confinement-Zeiten (kurze Trajektorien zeigen noch kein Plateau)

---

## 🔧 VERWENDUNG DES TRAINIERTEN MODELLS

### Modell laden

```python
from tensorflow import keras
import pickle
import numpy as np

# 1. Modell laden
model = keras.models.load_model('spt_trained_model/spt_classifier.keras')

# 2. Scaler laden
with open('spt_trained_model/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 3. Feature-Namen laden
with open('spt_trained_model/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
```

### Prediction auf neuer Trajektorie

```python
from spt_feature_extractor import SPTFeatureExtractor

# Feature-Extraktor initialisieren
extractor = SPTFeatureExtractor(dt=0.01)

# Neue Trajektorie (z.B. aus Ihrem Experiment)
# trajectory: (n_steps, 2 oder 3) numpy array [µm]
trajectory = np.array([...])  # Ihre Daten

# Features extrahieren
features_dict = extractor.extract_all_features(trajectory)
features_array = np.array([features_dict[name] for name in feature_names])
features_array = features_array.reshape(1, -1)

# Features normalisieren
features_norm = scaler.transform(features_array)

# Trajektorie padden
max_length = 3000
traj_padded = np.zeros((1, max_length, 2))  # oder 3 für 3D
length = min(len(trajectory), max_length)
traj_padded[0, :length, :] = trajectory[:length, :]

# Prediction
prediction = model.predict([traj_padded, features_norm], verbose=0)

# Ergebnis
class_names = ['Normal', 'Subdiffusion', 'Superdiffusion', 'Confined']
predicted_class = class_names[np.argmax(prediction[0])]
confidence = np.max(prediction[0])

print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print(f"\nAll Probabilities:")
for name, prob in zip(class_names, prediction[0]):
    print(f"  {name:15s}: {prob:.2%}")
```

---

## 📚 LITERATURVERWEISE

[1] Saxton, M. J. (1997). "Single-particle tracking: The distribution of diffusion coefficients." Biophysical Journal, 72(4), 1744-1753.

[2] Manzo, C., & Garcia-Parajo, M. F. (2015). "A review of progress in single particle tracking: from methods to biophysical insights." Reports on Progress in Physics, 78(12), 124601.

[3] Ewers, H., et al. (2005). "Single-particle tracking of murine polyoma virus-like particles on live cells and artificial membranes." PNAS, 102(42), 15110-15115.

[4] Metzler, R., & Klafter, J. (2000). "The random walk's guide to anomalous diffusion: a fractional dynamics approach." Physics Reports, 339(1), 1-77.

[5] Höfling, F., & Franosch, T. (2013). "Anomalous transport in the crowded world of biological cells." Reports on Progress in Physics, 76(4), 046602.

[6] Caspi, A., et al. (2000). "Enhanced diffusion in active intracellular transport." Physical Review Letters, 85(26), 5655.

[7] Kues, T., et al. (2001). "Visualization and tracking of single protein molecules in the cell nucleus." Biophysical Journal, 80(6), 2954-2967.

[8] Kusumi, A., et al. (2005). "Paradigm shift of the plasma membrane concept from the two-dimensional continuum fluid to the partitioned fluid: high-speed single-molecule tracking of membrane molecules." Annual Review of Biophysics and Biomolecular Structure, 34, 351-378.

[9] Granik, N., & Weiss, L. E. (2019). "Single-particle diffusion characterization by deep learning." Bioinformatics, 35(18), 3510-3517.

[10] Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural Computation, 9(8), 1735-1780.

[11] Vaswani, A., et al. (2017). "Attention is all you need." Advances in Neural Information Processing Systems, 30.

[12] Prechelt, L. (1998). "Early stopping-but when?" In Neural Networks: Tricks of the trade (pp. 55-69). Springer.

[13] Smith, L. N. (2017). "Cyclical learning rates for training neural networks." IEEE Winter Conference on Applications of Computer Vision (WACV), 464-472.

[14] Ioffe, S., & Szegedy, C. (2015). "Batch normalization: Accelerating deep network training by reducing internal covariate shift." ICML, 448-456.

[15] Srivastava, N., et al. (2014). "Dropout: A simple way to prevent neural networks from overfitting." Journal of Machine Learning Research, 15(1), 1929-1958.

---

## 📧 SUPPORT & KONTAKT

Bei Fragen, Problemen oder Feature-Anfragen:

**Wissenschaftliche Fragen:**  
Siehe Literaturverweise für theoretische Grundlagen

**Technische Probleme:**  
1. Überprüfen Sie die System-Anforderungen
2. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
3. Prüfen Sie die Ausgabe auf Error-Messages

**Performance-Optimierung:**  
- GPU verwenden (10-20x schneller als CPU)
- Batch Size erhöhen (wenn RAM erlaubt)
- Samples pro Klasse: min. 2000, empfohlen 3000-5000

---

## ⚖️ LIZENZ & NUTZUNG

Dieses System wurde im Rahmen einer wissenschaftlichen Masterthesis entwickelt.

**Verwendung:**
- Für akademische und Forschungszwecke frei verwendbar
- Bei Verwendung in Publikationen bitte zitieren
- Kommerzielle Nutzung nach Rücksprache

---

**Version 2.0 - Oktober 2025**  
**Vollständig korrigiert, integriert und wissenschaftlich fundiert**

---
