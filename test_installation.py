"""
QUICK TEST - SPT CLASSIFIER SYSTEM
===================================

Dieses Skript testet, ob alle Komponenten korrekt installiert sind.
Führe es aus, BEVOR du das vollständige Training startest!

Dauer: ~1 Minute
"""

import sys

print("="*80)
print("SPT CLASSIFIER - SYSTEM TEST")
print("="*80)
print("\n🔍 Überprüfe Installation...\n")

# Test 1: Python Version
print("1. Python Version:")
print(f"   Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ ERROR: Python >= 3.8 erforderlich!")
    sys.exit(1)
else:
    print("   ✅ OK")

# Test 2: Kern-Bibliotheken
print("\n2. Kern-Bibliotheken:")
try:
    import numpy as np
    print(f"   NumPy: {np.__version__} ✅")
except ImportError:
    print("   ❌ NumPy fehlt! Installiere: pip install numpy")
    sys.exit(1)

try:
    import scipy
    print(f"   SciPy: {scipy.__version__} ✅")
except ImportError:
    print("   ❌ SciPy fehlt! Installiere: pip install scipy")
    sys.exit(1)

try:
    import sklearn
    print(f"   Scikit-Learn: {sklearn.__version__} ✅")
except ImportError:
    print("   ❌ Scikit-Learn fehlt! Installiere: pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib
    print(f"   Matplotlib: {matplotlib.__version__} ✅")
except ImportError:
    print("   ❌ Matplotlib fehlt! Installiere: pip install matplotlib")
    sys.exit(1)

# Test 3: TensorFlow
print("\n3. TensorFlow:")
try:
    import tensorflow as tf
    print(f"   TensorFlow: {tf.__version__} ✅")
    
    # GPU-Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPU(s) gefunden: {len(gpus)} ✅")
        for i, gpu in enumerate(gpus):
            print(f"     - GPU {i}: {gpu.name}")
        print("   ⚡ Training wird SCHNELL sein!")
    else:
        print("   ⚠️  Keine GPU gefunden - Training auf CPU (langsamer)")
        print("      Für GPU-Support: pip install tensorflow[and-cuda]")
    
except ImportError:
    print("   ❌ TensorFlow fehlt! Installiere: pip install tensorflow")
    sys.exit(1)

# Test 4: Lokale Module
print("\n4. Lokale Module:")
try:
    from config_SPT import print_config_summary
    print("   config_SPT.py ✅")
except ImportError as e:
    print(f"   ❌ config_SPT.py fehlt! Error: {e}")
    sys.exit(1)

try:
    from spt_trajectory_generator import generate_normal_diffusion_spt
    print("   spt_trajectory_generator.py ✅")
except ImportError as e:
    print(f"   ❌ spt_trajectory_generator.py fehlt! Error: {e}")
    sys.exit(1)

try:
    from spt_feature_extractor import SPTFeatureExtractor
    print("   spt_feature_extractor.py ✅")
except ImportError as e:
    print(f"   ❌ spt_feature_extractor.py fehlt! Error: {e}")
    sys.exit(1)

try:
    from train_spt_classifier import SPTClassifierTrainer
    print("   train_spt_classifier.py ✅")
except ImportError as e:
    print(f"   ❌ train_spt_classifier.py fehlt! Error: {e}")
    sys.exit(1)

# Test 5: Funktions-Test
print("\n5. Funktions-Test:")
print("   Generiere Test-Trajektorie...")
try:
    test_traj = generate_normal_diffusion_spt(100, D=0.1, dimensionality='2D')
    print(f"   Shape: {test_traj.shape} ✅")
    
    extractor = SPTFeatureExtractor(dt=0.01)
    test_features = extractor.extract_all_features(test_traj)
    print(f"   Features extrahiert: {len(test_features)} ✅")
    
except Exception as e:
    print(f"   ❌ Fehler bei Funktions-Test: {e}")
    sys.exit(1)

# Test 6: Keras Model Build Test
print("\n6. Keras Model Test:")
print("   Baue kleines Test-Modell...")
try:
    from tensorflow.keras import layers, models
    
    # Mini-Modell
    inputs = layers.Input(shape=(100, 2))
    x = layers.Conv1D(32, 3)(inputs)
    x = layers.LSTM(16)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    test_model = models.Model(inputs, outputs)
    test_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print(f"   Model erstellt: {test_model.count_params()} Parameter ✅")
    
except Exception as e:
    print(f"   ❌ Fehler beim Model-Build: {e}")
    sys.exit(1)

# Abschluss
print("\n" + "="*80)
print("✅ ALLE TESTS BESTANDEN!")
print("="*80)
print("\n📋 ZUSAMMENFASSUNG:")
print(f"   Python: {sys.version.split()[0]}")
print(f"   TensorFlow: {tf.__version__}")
print(f"   GPU: {'Ja ⚡' if gpus else 'Nein (CPU)'}")
print(f"   Lokale Module: OK")
print("\n🚀 System bereit für Training!")
print("\nNächster Schritt:")
print("   python train_spt_classifier.py")
print("   ODER in Jupyter Notebook:")
print("   from train_spt_classifier import run_complete_training")
print("   run_complete_training()")
print("\n" + "="*80)
