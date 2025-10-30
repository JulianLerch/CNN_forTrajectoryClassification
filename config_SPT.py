"""
KONFIGURATION - SINGLE PARTICLE TRACKING (SPT) KLASSIFIKATION
==============================================================

Physikalisch korrekte Parameter fÃ¼r TDI-G0 Diffusion in Polymermatrizen

WISSENSCHAFTLICHE BASIS:
- TDI-G0: Terrylenediimid (Farbstoff, M_w â‰ˆ 600-800 g/mol)
- Polymer: Variable Polymerisationsgrade (stark â†’ Eduktschmelze)
- Technik: 2D/3D Single Particle Tracking (Fluoreszenz)

LITERATUR-REFERENZEN:
[1] Zettl et al. (2020) - TDI Diffusion in PMMA
[2] Papaleo et al. (2015) - Anomale Diffusion in Polymeren
[3] KÃ¶hler et al. (2021) - 3D SPT mit Astigmatismus
[4] Metzler & Klafter (2000) - Anomale Diffusion Theorie

Autor: Masterthesis TDI-G0 Diffusion
Datum: Oktober 2025
"""

import numpy as np

# =============================================================================
# 1. PHYSIKALISCHE KONSTANTEN
# =============================================================================

# Boltzmann-Konstante [J/K]
K_B = 1.380649e-23

# Temperatur [K] (typisch fÃ¼r SPT-Experimente)
TEMPERATURE = 298.15  # 25Â°C

# Thermal Energy [J]
K_B_T = K_B * TEMPERATURE  # â‰ˆ 4.11 Ã— 10â»Â²Â¹ J

# =============================================================================
# 2. TDI-G0 MOLEKÃœL-EIGENSCHAFTEN
# =============================================================================

# TDI-G0 Molekulargewicht [g/mol]
# Terrylenediimid mit Seitenketten
TDI_MOLECULAR_WEIGHT = 700  # Â±100 g/mol

# Hydrodynamischer Radius (AbschÃ¤tzung)
# FÃ¼r planares aromatisches MolekÃ¼l mit Seitenketten
TDI_HYDRODYNAMIC_RADIUS = 0.8e-9  # m (â‰ˆ 8 Ã…)

# =============================================================================
# 3. POLYMER-MATRIX EIGENSCHAFTEN
# =============================================================================

"""
Variable Polymerisationsgrade:

1. Stark polymerisiert (Glassy):
   - Hohe ViskositÃ¤t: Î· â‰ˆ 10Â³-10âµ PaÂ·s
   - EingeschrÃ¤nkte MobilitÃ¤t
   - D â‰ˆ 10â»â´ - 10â»Â² ÂµmÂ²/s
   - Oft subdiffusiv (Î± â‰ˆ 0.5-0.9)
   
2. Mittel polymerisiert:
   - Moderate ViskositÃ¤t: Î· â‰ˆ 10-10Â³ PaÂ·s
   - D â‰ˆ 10â»Â² - 10â»Â¹ ÂµmÂ²/s
   - Î± â‰ˆ 0.8-1.2
   
3. Schwach polymerisiert:
   - Niedrige ViskositÃ¤t: Î· â‰ˆ 1-10 PaÂ·s
   - D â‰ˆ 10â»Â¹ - 1 ÂµmÂ²/s
   - NÃ¤her an normaler Diffusion
   
4. Eduktschmelze (Monomer):
   - FlÃ¼ssig: Î· â‰ˆ 0.01-1 PaÂ·s
   - D â‰ˆ 0.5-5 ÂµmÂ²/s
   - Normal bis leicht superdiffusiv
"""

# Diffusionskoeffizient-Bereiche [ÂµmÂ²/s]
# Basierend auf Stokes-Einstein: D = k_BÂ·T/(6Ï€Î·r)

D_RANGES = {
    'glassy': (1e-4, 1e-2),      # Stark polymerisiert
    'moderate': (1e-2, 1e-1),    # Mittel polymerisiert  
    'weak': (1e-1, 1.0),         # Schwach polymerisiert
    'monomer': (0.5, 5.0)        # Eduktschmelze
}

# FÃ¼r TRAINING: Gesamtbereich abdecken
D_MIN_2D = 1e-4  # ÂµmÂ²/s
D_MAX_2D = 3.0   # ÂµmÂ²/s

D_MIN_3D = 1e-4  # ÂµmÂ²/s  
D_MAX_3D = 3.0   # ÂµmÂ²/s

# =============================================================================
# 4. ANOMALE DIFFUSIONS-PARAMETER
# =============================================================================

# Anomaler Exponent Î± (dimensionslos)
ALPHA_RANGES = {
    'normal': (0.95, 1.05),           # Quasi-normal (enge Toleranz!)
    'subdiffusion': (0.3, 0.9),       # Subdiffusiv
    'superdiffusion': (1.1, 1.8),     # Superdiffusiv
    'confined': (0.7, 1.05)           # Kann normal ODER subdiffusiv sein
}

# Generalisierter Diffusionskoeffizient D_Î± [ÂµmÂ²/s^Î±]
# FÃ¼r Î± â‰  1: D_Î± ist nicht gleich D!
D_ALPHA_RANGES = {
    'subdiffusion': (1e-3, 0.5),      # ÂµmÂ²/s^Î±
    'superdiffusion': (1e-2, 1.0)     # ÂµmÂ²/s^Î±
}

# =============================================================================
# 5. CONFINEMENT-PARAMETER
# =============================================================================

"""
Confinement in Polymermatrizen:

1. Mesh Size (Polymernetzwerk):
   - Stark polymerisiert: Î¾ â‰ˆ 1-10 nm
   - Mittel: Î¾ â‰ˆ 10-100 nm
   - Schwach: Î¾ â‰ˆ 100-1000 nm

2. DomÃ¤nen/Phasen-Separation:
   - Typisch: 100-500 nm

3. FÃ¼r SPT sichtbar: R_conf > 100 nm (Diffraktionslimit!)
"""

# Confinement-Radien [Âµm]
CONFINEMENT_RADIUS_MIN = 0.1   # 100 nm (Diffraktionslimit)
CONFINEMENT_RADIUS_MAX = 2.0   # 2 Âµm (groÃŸe DomÃ¤ne)

# Geometrien
CONFINEMENT_GEOMETRIES = ['circular', 'square', 'elliptical']

# =============================================================================
# 6. EXPERIMENTELLE PARAMETER (SPT)
# =============================================================================

# ZeitauflÃ¶sung [s]
DT_MIN = 0.001   # 1 ms (schnelles Tracking)
DT_TYPICAL = 0.01  # 10 ms (Standard)
DT_MAX = 0.1     # 100 ms (langsames Tracking)

# FÃ¼r Training
DT = 0.01  # s (10 ms frame rate)

# Trajektorien-LÃ¤ngen [Frames]
TRAJECTORY_LENGTH_MIN = 30      # Minimum fÃ¼r MSD-Fit
TRAJECTORY_LENGTH_MAX = 5000    # Lange Tracks fÃ¼r Statistik

# Lokalisierungs-Precision [Âµm]
# AbhÃ¤ngig von Photon-Zahl, Pixel-GrÃ¶ÃŸe, etc.
LOCALIZATION_PRECISION_2D = 0.015  # Âµm (â‰ˆ15 nm, typisch fÃ¼r gute SNR)
LOCALIZATION_PRECISION_Z = 0.050   # Âµm (â‰ˆ50 nm, schlechter in z)

# Astigmatismus-Parameter fÃ¼r 3D
# z-Position aus PSF-ElliptizitÃ¤t
ASTIGMATISM_CALIBRATION = {
    'focal_length': 2.0,    # Âµm (Bereich wo PSF elliptisch)
    'z_precision': 0.05     # Âµm (axiale Lokalisierung)
}

# =============================================================================
# 7. DATENGENERIERUNGS-PARAMETER
# =============================================================================

# Anzahl Samples pro Klasse (ERHÖHT für bessere Generalisierung!)
N_SAMPLES_PER_CLASS = 10000  # War 5000, jetzt 10000 für robusteres Training

# Class Balance
CLASS_WEIGHTS = {
    'normal': 2.0,          # Boost Normal (wichtigste Baseline)
    'subdiffusion': 1.5,    # Boost Subdiffusion (hÃ¤ufig in Polymer)
    'superdiffusion': 1.0,  # Standard
    'confined': 2.0         # Boost Confined (oft schwach)
}

# 2D vs 3D Ratio
RATIO_2D = 0.5  # 50% 2D Trajektorien
RATIO_3D = 0.5  # 50% 3D Trajektorien

# =============================================================================
# 8. NEURAL NETWORK ARCHITEKTUR
# =============================================================================

# Feature-Dimensionen
N_FEATURES_2D = 15  # 15 Features fÃ¼r 2D
N_FEATURES_3D = 18  # 18 Features fÃ¼r 3D (+ 3 z-spezifische)

# Netzwerk-GrÃ¶ÃŸe
NN_ARCHITECTURE = {
    'input_dim': max(N_FEATURES_2D, N_FEATURES_3D),
    'hidden_layers': [256, 512, 512, 256, 128],  # Tiefes Netzwerk!
    'dropout': 0.3,
    'use_batch_norm': True,
    'use_attention': True,  # Self-Attention Layer
    'use_residual': True    # Residual Connections
}

# Training (OPTIMIERT für schnelleres & besseres Training!)
TRAINING_CONFIG = {
    'batch_size': 512,       # Erhöht von 256 -> schnelleres Training
    'epochs': 150,           # Erhöht von 100 -> mehr Zeit für Konvergenz
    'learning_rate': 3e-4,   # Erhöht von 1e-4 -> schnellere Konvergenz
    'optimizer': 'adamw',
    'loss': 'categorical_crossentropy',
    'early_stopping_patience': 20,  # Erhöht von 15 -> mehr Geduld
    'reduce_lr_patience': 7,    # Reduziert von 8 -> schnellere LR-Anpassung
    'min_lr': 1e-7,
    'reduce_lr_factor': 0.3  # NEU: Aggressivere LR-Reduktion
}

# Target Performance
TARGET_METRICS = {
    'overall_accuracy': 0.97,      # 97% Gesamt
    'per_class_accuracy': 0.95,    # 95% pro Klasse
    'per_class_f1': 0.94           # F1-Score â‰¥ 94%
}

# =============================================================================
# 9. FEATURE EXTRACTION CONFIG
# =============================================================================

FEATURE_CONFIG = {
    # MSD-Fitting
    'msd_max_lag_fraction': 0.25,  # Bis 25% der Trajektorien-LÃ¤nge
    'msd_min_lags': 10,            # Minimum Lags fÃ¼r Fit
    
    # Velocity Autocorrelation
    'velocity_max_lag': 10,        # Lags fÃ¼r Autokorrelation
    
    # Confinement Detection
    'confinement_test_radius': 3,  # Vielfaches von Rg
    
    # Gaussianity
    'gaussianity_lag': 1,          # Lag fÃ¼r Gaussianity-Test
    
    # 3D-spezifisch
    'z_anisotropy_threshold': 0.3  # Schwelle fÃ¼r z-Anisotropie
}

# =============================================================================
# 10. VALIDIERUNG & EXPORT
# =============================================================================

VALIDATION_CONFIG = {
    'test_split': 0.2,             # 20% Test-Daten
    'stratified': True,            # Stratifiziert nach Klasse
    'cross_validation_folds': 5,   # 5-Fold CV
    'confusion_matrix': True,
    'roc_curves': True,
    'per_length_analysis': True    # Analyse nach Trajektorien-LÃ¤nge
}

OUTPUT_CONFIG = {
    'model_format': 'keras',
    'save_scaler': True,
    'save_feature_names': True,
    'save_training_history': True,
    'export_onnx': False  # Optional: ONNX fÃ¼r andere Frameworks
}

# =============================================================================
# 11. REPRODUZIERBARKEIT
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# 12. WISSENSCHAFTLICHE VALIDIERUNG
# =============================================================================

"""
Validierungs-Checks fÃ¼r physikalische PlausibilitÃ¤t:

1. Einstein-Smoluchowski Check:
   2D: <rÂ²> = 4Dt  â†’  D = <rÂ²>/(4t)
   3D: <rÂ²> = 6Dt  â†’  D = <rÂ²>/(6t)

2. MSD-LinearitÃ¤t (log-log):
   log(MSD) = log(4D_Î±) + Î±Â·log(t)
   â†’ Slope = Î±

3. Confined Plateau:
   MSD(t â†’ âˆž) â†’ const â‰ˆ R_confÂ²

4. Lokalisierungs-Precision:
   Ïƒ_loc << âˆš(4Dt) fÃ¼r normale Tracks
   (sonst: static localization error dominiert)
"""

VALIDATION_CHECKS = {
    'check_msd_slope': True,
    'check_d_range': True,
    'check_confinement_plateau': True,
    'check_localization_ratio': True
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_d_range_for_polymer_state(state='moderate'):
    """
    Gibt realistischen D-Bereich fÃ¼r Polymerisationszustand zurÃ¼ck
    
    Args:
        state: 'glassy', 'moderate', 'weak', 'monomer'
    
    Returns:
        (D_min, D_max) in ÂµmÂ²/s
    """
    return D_RANGES.get(state, D_RANGES['moderate'])


def estimate_viscosity_from_d(D, r=TDI_HYDRODYNAMIC_RADIUS, T=TEMPERATURE):
    """
    SchÃ¤tzt ViskositÃ¤t aus Diffusionskoeffizient (Stokes-Einstein)
    
    Args:
        D: Diffusionskoeffizient [ÂµmÂ²/s]
        r: Hydrodynamischer Radius [m]
        T: Temperatur [K]
    
    Returns:
        Î·: ViskositÃ¤t [PaÂ·s]
    """
    D_si = D * 1e-12  # ÂµmÂ²/s â†’ mÂ²/s
    k_b_t = K_B * T
    eta = k_b_t / (6 * np.pi * r * D_si)
    return eta


def get_confinement_time(R_conf, D):
    """
    Charakteristische Confinement-Zeit
    
    Args:
        R_conf: Confinement-Radius [Âµm]
        D: Diffusionskoeffizient [ÂµmÂ²/s]
    
    Returns:
        Ï„_conf: Zeit [s]
    """
    return R_conf**2 / (4 * D)


def print_config_summary():
    """Gibt Konfigurations-Zusammenfassung aus"""
    print("="*80)
    print("SPT KLASSIFIKATION - KONFIGURATION")
    print("="*80)
    print(f"\nMOLEKÃœL: TDI-G0")
    print(f"  Molekulargewicht: {TDI_MOLECULAR_WEIGHT} g/mol")
    print(f"  Hydrodynamischer Radius: {TDI_HYDRODYNAMIC_RADIUS*1e9:.1f} Ã…")
    
    print(f"\nPHYSIKALISCHE PARAMETER:")
    print(f"  Temperatur: {TEMPERATURE} K ({TEMPERATURE-273.15}Â°C)")
    print(f"  k_BÂ·T: {K_B_T:.2e} J")
    
    print(f"\nDIFFUSIONSKOEFFIZIENT-BEREICHE:")
    print(f"  2D: {D_MIN_2D:.1e} - {D_MAX_2D:.1f} ÂµmÂ²/s")
    print(f"  3D: {D_MIN_3D:.1e} - {D_MAX_3D:.1f} ÂµmÂ²/s")
    
    print(f"\nANOMALE EXPONENTEN:")
    for key, val in ALPHA_RANGES.items():
        print(f"  {key:15s}: Î± = {val[0]:.2f} - {val[1]:.2f}")
    
    print(f"\nCONFINEMENT:")
    print(f"  Radius: {CONFINEMENT_RADIUS_MIN:.2f} - {CONFINEMENT_RADIUS_MAX:.2f} Âµm")
    print(f"  Geometrien: {', '.join(CONFINEMENT_GEOMETRIES)}")
    
    print(f"\nEXPERIMENTELL:")
    print(f"  Zeitschritt dt: {DT*1000:.1f} ms")
    print(f"  Trajektorien-LÃ¤nge: {TRAJECTORY_LENGTH_MIN} - {TRAJECTORY_LENGTH_MAX} Frames")
    print(f"  Lokalisierungs-Precision xy: {LOCALIZATION_PRECISION_2D*1000:.1f} nm")
    print(f"  Lokalisierungs-Precision z: {LOCALIZATION_PRECISION_Z*1000:.1f} nm")
    
    print(f"\nTRAINING:")
    print(f"  Samples pro Klasse: {N_SAMPLES_PER_CLASS}")
    print(f"  2D/3D Ratio: {RATIO_2D:.0%} / {RATIO_3D:.0%}")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    
    print(f"\nZIEL-METRIKEN:")
    print(f"  Overall Accuracy: â‰¥{TARGET_METRICS['overall_accuracy']:.1%}")
    print(f"  Per-Class Accuracy: â‰¥{TARGET_METRICS['per_class_accuracy']:.1%}")
    print(f"  Per-Class F1: â‰¥{TARGET_METRICS['per_class_f1']:.1%}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print_config_summary()
    
    # Beispiel-Berechnungen
    print("\n" + "="*80)
    print("BEISPIEL-BERECHNUNGEN")
    print("="*80)
    
    # Verschiedene Polymer-ZustÃ¤nde
    states = ['glassy', 'moderate', 'weak', 'monomer']
    
    print("\nGeschÃ¤tzte ViskositÃ¤ten (Stokes-Einstein):")
    print(f"{'Zustand':<15} {'D [ÂµmÂ²/s]':<15} {'Î· [PaÂ·s]':<15}")
    print("-"*45)
    
    for state in states:
        D_min, D_max = get_d_range_for_polymer_state(state)
        D_mid = np.sqrt(D_min * D_max)  # Geometrisches Mittel
        eta = estimate_viscosity_from_d(D_mid)
        print(f"{state:<15} {D_mid:.2e}        {eta:.2e}")
    
    print("\nConfinement-Zeiten (fÃ¼r R=1Âµm):")
    print(f"{'D [ÂµmÂ²/s]':<15} {'Ï„_conf [s]':<15}")
    print("-"*30)
    
    for D in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        tau = get_confinement_time(1.0, D)
        print(f"{D:.1e}        {tau:.2f}")
    
    print("\n" + "="*80)
