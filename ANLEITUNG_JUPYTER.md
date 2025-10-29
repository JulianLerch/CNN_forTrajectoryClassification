# JUPYTER NOTEBOOK ANLEITUNG - SPT CLASSIFIER
## Schritt-für-Schritt Anleitung für Anfänger

---

## 📘 WAS IST JUPYTER NOTEBOOK?

Jupyter Notebook ist eine interaktive Programmier-Umgebung, die es erlaubt, Code in einzelnen "Zellen" auszuführen. Perfekt zum Experimentieren und Lernen!

**Vorteile:**
- Code-Blöcke einzeln ausführen
- Sofortige Visualisierung von Plots
- Markdown für Notizen
- Wiederholbare Analysen

---

## 🚀 INSTALLATION

### Schritt 1: Anaconda installieren (empfohlen)

1. Download Anaconda: https://www.anaconda.com/download
2. Installieren (alle Standardeinstellungen akzeptieren)
3. Öffne "Anaconda Navigator"

### Schritt 2: Jupyter Notebook starten

**Option A: Über Anaconda Navigator:**
1. Öffne Anaconda Navigator
2. Klicke auf "Launch" unter Jupyter Notebook
3. Browser öffnet sich automatisch

**Option B: Über Kommandozeile:**
```bash
jupyter notebook
```

---

## 📂 SETUP IM NOTEBOOK

### Schritt 1: Neues Notebook erstellen

1. Navigiere im Browser zum Projektordner
2. Klicke auf "New" → "Python 3"
3. Notebook öffnet sich

### Schritt 2: Bibliotheken importieren

Führe diese Zelle **ZUERST** aus (Shift+Enter):

```python
# Cell 1: Imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Lokale Module (müssen im gleichen Ordner sein!)
from train_spt_classifier import SPTClassifierTrainer

print("✅ Imports erfolgreich!")
print("Bereit zum Training.")
```

---

## 🎓 KOMPLETTES TRAINING IN JUPYTER

### Option A: Automatisches Training (Einfachste Methode)

```python
# Cell 2: Automatisches Training

from train_spt_classifier import run_complete_training

# Konfiguration
CONFIG = {
    'n_samples_per_class': 2000,  # Weniger Samples = schneller (für Test)
    'dimensionality': '2D',       # '2D' oder '3D'
    'polymerization_degree': 0.5, # 0.0-1.0
    'epochs': 50,                 # Weniger Epochen für schnelles Testen
    'batch_size': 32,
    'output_dir': './mein_erstes_modell'
}

# Training starten
trainer = run_complete_training(**CONFIG)

print("\n✅ Training fertig!")
print(f"Modell gespeichert in: {CONFIG['output_dir']}/")
```

**Dauer:** Ca. 20-30 Minuten (je nach Hardware)

---

### Option B: Schritt-für-Schritt (Lernmodus)

Perfekt um jeden Schritt zu verstehen und anzupassen!

#### Schritt 1: Trainer initialisieren

```python
# Cell 3: Trainer initialisieren

trainer = SPTClassifierTrainer(
    max_length=3000,              # Max Länge für Trajektorien
    n_features=24,                # 24 physikalische Features
    output_dir='./mein_modell',   # Wo das Modell gespeichert wird
    random_seed=42                # Für Reproduzierbarkeit
)

print("✅ Trainer initialisiert")
```

#### Schritt 2: Daten generieren

```python
# Cell 4: Datengenerierung

# WICHTIG: Das kann 5-10 Minuten dauern!
trainer.generate_training_data(
    n_samples_per_class=2000,     # Anzahl Samples pro Klasse
    dimensionality='2D',          # '2D' oder '3D'
    polymerization_degree=0.5,    # Polymer-Zustand (0.0-1.0)
    verbose=True                  # Zeige Fortschritt
)

print("\n✅ Daten generiert und vorbereitet")
print(f"Train-Set: {len(trainer.X_traj_train)} Samples")
print(f"Val-Set: {len(trainer.X_traj_val)} Samples")
print(f"Test-Set: {len(trainer.X_traj_test)} Samples")
```

**Was passiert hier?**
1. Generiert synthetische Trajektorien (4 Klassen)
2. Extrahiert 24 Features pro Trajektorie
3. Teilt Daten auf: 70% Train, 15% Validation, 15% Test
4. Standardisiert Features (Mittelwert=0, Standardabweichung=1)

#### Schritt 3: Modell bauen

```python
# Cell 5: Modell-Konstruktion

trainer.build_model(verbose=True)

print("\n✅ Modell konstruiert")
print(f"Total Parameter: {trainer.model.count_params():,}")
```

**Architektur:**
- Trajectory Branch: CNN + LSTM + Attention
- Feature Branch: Dense Layers
- Fusion: Concat + Classification

#### Schritt 4: Training

```python
# Cell 6: Model Training

# WARNUNG: Das dauert 20-40 Minuten!
trainer.train(
    epochs=50,        # Max Epochen (Early Stopping aktiv)
    batch_size=32,    # Batch Size (höher = schneller, braucht mehr RAM)
    verbose=1         # Zeige Fortschritt
)

print("\n✅ Training abgeschlossen")
```

**Was passiert hier?**
- Training läuft Epoche für Epoche
- Validierungs-Loss wird überwacht
- Bei Plateau: Learning Rate wird reduziert
- Bei zu langem Plateau: Training stoppt (Early Stopping)
- Bestes Modell wird automatisch gespeichert

**Output:**
- Fortschrittsbalken pro Epoche
- Loss und Accuracy für Train und Validation
- Automatische Plots am Ende

#### Schritt 5: Evaluation

```python
# Cell 7: Model Evaluation

accuracy, report, cm = trainer.evaluate(verbose=True)

print(f"\n🎯 Test Accuracy: {accuracy:.2%}")
```

**Output:**
- Confusion Matrix (Visualisierung)
- Classification Report (Precision, Recall, F1)
- Per-Class Performance

#### Schritt 6: Speichern

```python
# Cell 8: Model speichern

trainer.save_model(verbose=True)

print("\n✅ Alle Dateien gespeichert!")
```

**Gespeicherte Dateien:**
- `spt_classifier.keras` - Trainiertes Modell
- `feature_scaler.pkl` - Für Feature-Normalisierung
- `feature_names.pkl` - Namen der Features
- `metadata.json` - Konfiguration
- Plots als PNG

---

## 📊 VISUALISIERUNGEN

### Beispiel-Trajektorien anschauen

```python
# Cell 9: Visualisierung

from spt_trajectory_generator import (
    generate_normal_diffusion_spt,
    generate_subdiffusion_spt,
    generate_superdiffusion_spt,
    generate_confined_diffusion_spt
)

# Generiere Beispiel-Trajektorien
traj_normal = generate_normal_diffusion_spt(300, D=0.1, dimensionality='2D')
traj_sub = generate_subdiffusion_spt(300, alpha=0.5, D_alpha=0.08, dimensionality='2D')
traj_super = generate_superdiffusion_spt(300, alpha=1.6, D_alpha=0.12, dimensionality='2D')
traj_conf = generate_confined_diffusion_spt(300, confinement_radius=1.0, D=0.08, dimensionality='2D')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Normal
axes[0,0].plot(traj_normal[:,0], traj_normal[:,1], 'o-', ms=2, alpha=0.7, color='green')
axes[0,0].plot(traj_normal[0,0], traj_normal[0,1], 'go', ms=10, label='Start')
axes[0,0].plot(traj_normal[-1,0], traj_normal[-1,1], 'ro', ms=10, label='End')
axes[0,0].set_title('Normal Diffusion (α=1.0)', fontweight='bold', fontsize=14)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axis('equal')

# Subdiffusion
axes[0,1].plot(traj_sub[:,0], traj_sub[:,1], 'o-', ms=2, alpha=0.7, color='red')
axes[0,1].plot(traj_sub[0,0], traj_sub[0,1], 'go', ms=10)
axes[0,1].plot(traj_sub[-1,0], traj_sub[-1,1], 'ro', ms=10)
axes[0,1].set_title('Subdiffusion (α=0.5)', fontweight='bold', fontsize=14)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].axis('equal')

# Superdiffusion
axes[1,0].plot(traj_super[:,0], traj_super[:,1], 'o-', ms=2, alpha=0.7, color='blue')
axes[1,0].plot(traj_super[0,0], traj_super[0,1], 'go', ms=10)
axes[1,0].plot(traj_super[-1,0], traj_super[-1,1], 'ro', ms=10)
axes[1,0].set_title('Superdiffusion (α=1.6)', fontweight='bold', fontsize=14)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axis('equal')

# Confined
axes[1,1].plot(traj_conf[:,0], traj_conf[:,1], 'o-', ms=2, alpha=0.7, color='orange')
axes[1,1].plot(traj_conf[0,0], traj_conf[0,1], 'go', ms=10)
axes[1,1].plot(traj_conf[-1,0], traj_conf[-1,1], 'ro', ms=10)
circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', linewidth=2)
axes[1,1].add_patch(circle)
axes[1,1].set_title('Confined Diffusion (R=1.0 µm)', fontweight='bold', fontsize=14)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].axis('equal')

for ax in axes.flat:
    ax.set_xlabel('x [µm]', fontsize=12)
    ax.set_ylabel('y [µm]', fontsize=12)

plt.tight_layout()
plt.show()

print("✅ Beispiel-Trajektorien visualisiert")
```

---

## 🔍 MODELL VERWENDEN (INFERENCE)

### Eigene Trajektorie klassifizieren

```python
# Cell 10: Inference auf neuer Trajektorie

from tensorflow import keras
import pickle
from spt_feature_extractor import SPTFeatureExtractor

# Modell und Scaler laden
model = keras.models.load_model('./mein_modell/spt_classifier.keras')
with open('./mein_modell/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./mein_modell/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Feature-Extraktor
extractor = SPTFeatureExtractor(dt=0.01)

# BEISPIEL: Neue Trajektorie (ersetze mit deinen Daten!)
# trajectory sollte shape (n_steps, 2) oder (n_steps, 3) haben [µm]
trajectory = generate_normal_diffusion_spt(200, D=0.1, dimensionality='2D')

# Features extrahieren
features_dict = extractor.extract_all_features(trajectory)
features_array = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
features_norm = scaler.transform(features_array)

# Trajektorie padden
max_length = 3000
traj_padded = np.zeros((1, max_length, 2))
length = min(len(trajectory), max_length)
traj_padded[0, :length, :] = trajectory[:length, :]

# Prediction
prediction = model.predict([traj_padded, features_norm], verbose=0)

# Ergebnis
class_names = ['Normal', 'Subdiffusion', 'Superdiffusion', 'Confined']
predicted_class = class_names[np.argmax(prediction[0])]
confidence = np.max(prediction[0])

print(f"\n🎯 PREDICTION:")
print(f"Klasse: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print(f"\nAlle Wahrscheinlichkeiten:")
for name, prob in zip(class_names, prediction[0]):
    print(f"  {name:15s}: {prob*100:5.2f}%")

# Visualisiere Trajektorie
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:,0], trajectory[:,1], 'o-', ms=3, alpha=0.7)
plt.plot(trajectory[0,0], trajectory[0,1], 'go', ms=12, label='Start')
plt.plot(trajectory[-1,0], trajectory[-1,1], 'ro', ms=12, label='End')
plt.xlabel('x [µm]', fontsize=12)
plt.ylabel('y [µm]', fontsize=12)
plt.title(f'Predicted: {predicted_class} ({confidence:.1%})', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

---

## ⚙️ PARAMETER ANPASSEN

### Wichtige Hyperparameter

**Daten:**
```python
n_samples_per_class = 2000   # 2000-5000 (mehr = besser, aber langsamer)
dimensionality = '2D'        # '2D' oder '3D'
polymerization_degree = 0.5  # 0.0 (Monomer) bis 1.0 (voll polymerisiert)
```

**Training:**
```python
epochs = 50                  # 50-150 (Early Stopping stoppt automatisch)
batch_size = 32              # 16, 32, 64 (höher = schneller, braucht mehr RAM)
```

**Diffusionsbereiche ändern (in config_SPT.py):**
```python
D_MIN_2D = 1e-4  # Minimaler D-Wert [µm²/s]
D_MAX_2D = 3.0   # Maximaler D-Wert [µm²/s]
```

---

## 🐛 HÄUFIGE PROBLEME & LÖSUNGEN

### Problem 1: "ModuleNotFoundError"

**Ursache:** Dateien nicht im gleichen Ordner oder Bibliothek fehlt

**Lösung:**
```python
# Installiere fehlende Bibliothek
!pip install tensorflow numpy scipy scikit-learn matplotlib seaborn

# Überprüfe Dateien
import os
print("Dateien im Ordner:")
print(os.listdir('.'))
```

### Problem 2: "Out of Memory" (OOM)

**Ursache:** Zu wenig RAM

**Lösung:**
```python
# Reduziere Batch Size
batch_size = 16  # statt 32

# Oder: Reduziere Samples
n_samples_per_class = 1000  # statt 3000
```

### Problem 3: Training dauert ewig

**Ursache:** Keine GPU oder CPU zu langsam

**Lösung:**
```python
# Reduziere Komplexität für Tests:
n_samples_per_class = 500   # Weniger Daten
epochs = 20                 # Weniger Epochen
max_length = 1000           # Kürzere Trajektorien

# Oder: GPU aktivieren (wenn vorhanden)
import tensorflow as tf
print("GPU verfügbar:", tf.config.list_physical_devices('GPU'))
```

### Problem 4: "ValueError: dimensions mismatch"

**Ursache:** Trajektorie hat falsche Shape

**Lösung:**
```python
# Trajektorie muss shape (n_steps, 2) oder (n_steps, 3) haben
trajectory = np.array([[x1, y1], [x2, y2], ...])  # 2D
# ODER
trajectory = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # 3D

# Überprüfe Shape
print(f"Trajectory Shape: {trajectory.shape}")  # Sollte (n, 2) oder (n, 3) sein
```

---

## 💡 TIPPS & TRICKS

### Tipp 1: Schnelles Prototyping

Für erste Tests:
```python
CONFIG = {
    'n_samples_per_class': 500,   # Klein für schnellen Test
    'epochs': 20,
    'batch_size': 32
}
```

### Tipp 2: GPU-Beschleunigung

Wenn GPU verfügbar:
```python
import tensorflow as tf

# Zeige verfügbare GPUs
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Force GPU usage
with tf.device('/GPU:0'):
    trainer.train(epochs=100, batch_size=64)
```

### Tipp 3: Fortschritt speichern

Während Training:
```python
# Training History wird automatisch gespeichert
# Nach jedem Training abrufbar:
history = trainer.history.history

# Plot manuell erstellen
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()
```

### Tipp 4: Mehrere Modelle vergleichen

```python
# Trainiere Modelle mit verschiedenen Configs
configs = [
    {'n_samples': 1000, 'poly_deg': 0.3, 'name': 'model_low_poly'},
    {'n_samples': 1000, 'poly_deg': 0.7, 'name': 'model_high_poly'},
]

results = []
for cfg in configs:
    trainer = SPTClassifierTrainer(output_dir=f"./{cfg['name']}")
    trainer.generate_training_data(
        n_samples_per_class=cfg['n_samples'],
        polymerization_degree=cfg['poly_deg']
    )
    trainer.build_model()
    trainer.train(epochs=50, batch_size=32)
    acc, _, _ = trainer.evaluate()
    results.append({'name': cfg['name'], 'accuracy': acc})
    trainer.save_model()

# Vergleiche Resultate
for res in results:
    print(f"{res['name']}: {res['accuracy']:.2%}")
```

---

## 📚 WEITERFÜHRENDE THEMEN

### 1. 3D Training

```python
# Ändere einfach dimensionality auf '3D'
trainer.generate_training_data(
    n_samples_per_class=2000,
    dimensionality='3D',  # <-- HIER
    polymerization_degree=0.5
)
```

**Unterschiede:**
- Trajektorien haben 3 Dimensionen (x, y, z)
- 24 Features statt 15 (zusätzliche z-Features)
- Längere Trainingszeit (~30% mehr)

### 2. Transfer Learning

```python
# Lade vortrainiertes Modell
base_model = keras.models.load_model('./pretrained_model/spt_classifier.keras')

# Freeze erste Layer
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Fine-tune auf neuen Daten
# ... (deine neuen Daten)
```

### 3. Hyperparameter-Optimierung

```python
# Grid Search über verschiedene Configs
param_grid = {
    'batch_size': [16, 32, 64],
    'learning_rate': [1e-4, 5e-4, 1e-3],
}

best_acc = 0
best_params = None

for bs in param_grid['batch_size']:
    for lr in param_grid['learning_rate']:
        # Trainiere Modell...
        # Evaluiere...
        # Speichere beste Config
```

---

## ✅ CHECKLISTE

Vor dem Start:
- [ ] Anaconda installiert
- [ ] Jupyter Notebook läuft
- [ ] Alle Dateien im gleichen Ordner
- [ ] TensorFlow installiert (`pip install tensorflow`)

Während Training:
- [ ] Cell-by-Cell ausführen (nicht alles auf einmal!)
- [ ] Output beobachten (Errors? Warnings?)
- [ ] Plots anschauen
- [ ] Patience haben (Training dauert!)

Nach Training:
- [ ] Modell gespeichert?
- [ ] Accuracy > 90%?
- [ ] Confusion Matrix sinnvoll?
- [ ] Test auf eigenen Daten

---

## 🎓 ZUSAMMENFASSUNG

**Minimaler Workflow:**
1. Imports
2. `run_complete_training(...)` ausführen
3. Warten (20-40 Min)
4. Fertig! 🎉

**Lern-Workflow:**
1. Trainer initialisieren
2. Daten generieren
3. Modell bauen
4. Training
5. Evaluation
6. Speichern

**Inference-Workflow:**
1. Modell laden
2. Features extrahieren
3. Prediction
4. Interpretieren

---

**Viel Erfolg beim Training! 🚀**

Bei Problemen: Siehe "Häufige Probleme & Lösungen" oben.

---
