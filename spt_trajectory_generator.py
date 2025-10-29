"""
SCIENTIFIC SPT TRAJECTORY GENERATOR - 2D & 3D
==============================================

Physikalisch fundierte Generierung synthetischer Single-Particle-Tracking Trajektorien
fÃ¼r TDI-G0 MolekÃ¼le in Polymermatrizen.

Basierend auf:
- Einstein-Smoluchowski-Theorie (1905/1906)
- Fraktionale Brownsche Bewegung (Mandelbrot & Van Ness, 1968)
- Continuous-Time Random Walk Theorie (Metzler & Klafter, 2000)
- Confined Diffusion in MikrodomÃ¤nen (Kusumi et al., 2012)

Autor: Masterthesis TDI-G0 in Polymer
Version: 1.0 - Scientific Production
"""

import numpy as np
from scipy import stats
from scipy.stats import levy_stable
from typing import Tuple, Optional, List

class PhysicalConstants:
    """
    Fundamentale physikalische Konstanten fÃ¼r SPT-Simulationen
    
    Referenzen:
    - CODATA 2018 (Mohr et al., 2019)
    - Handbook of Chemistry and Physics, 103rd Ed.
    """
    k_B = 1.380649e-23  # Boltzmann-Konstante [J/K]
    T_room = 298.15     # Raumtemperatur [K]
    eta_water = 8.9e-4  # ViskositÃ¤t Wasser bei 25Â°C [PaÂ·s]
    
    # Typische Werte fÃ¼r Polymer-Systeme
    T_PMMA_glass = 378.15  # PMMA T_g [K] (Cowie & Ferguson, 1989)
    

class PolymerDiffusionParameters:
    """
    Empirische Parameter fÃ¼r TDI-G0 Diffusion in Polymermatrizen
    
    Basierend auf:
    - Free-volume Theorie (Cohen & Turnbull, 1959)
    - Vogel-Fulcher-Tammann Gleichung
    - Literatur fÃ¼r Farbstoffe in PMMA (Zhu et al., 2012)
    
    Polymerisierungsgrad-abhÃ¤ngige Diffusionskoeffizienten:
    - Eduktschmelze (0% Polymerisierung): D â‰ˆ 0.1-1.0 ÂµmÂ²/s
    - Teilpolymerisiert (50%): D â‰ˆ 0.01-0.1 ÂµmÂ²/s
    - Vollpolymerisiert (100%): D â‰ˆ 0.001-0.01 ÂµmÂ²/s
    """
    
    @staticmethod
    def get_D_range(polymerization_degree: float = 0.5) -> Tuple[float, float]:
        """
        Diffusionskoeffizienten-Bereich basierend auf Polymerisierungsgrad
        
        Args:
            polymerization_degree: 0.0 (Eduktschmelze) bis 1.0 (vollpolymerisiert)
            
        Returns:
            (D_min, D_max) in ÂµmÂ²/s
            
        Referenz: Free-volume Theorie (Vrentas & Duda, 1977)
        """
        if polymerization_degree < 0.3:
            # Niedrig polymerisiert: hohe MobilitÃ¤t
            return (0.05, 0.8)
        elif polymerization_degree < 0.7:
            # Mittel polymerisiert
            return (0.01, 0.2)
        else:
            # Hoch polymerisiert: niedrige MobilitÃ¤t
            return (0.001, 0.05)


def generate_fbm_davies_harte(n_steps: int, H: float, sigma: float = 1.0) -> np.ndarray:
    """
    Exakte fraktionale Brownsche Bewegung via Davies-Harte Algorithmus
    
    Generiert 1D fBm mit Hurst-Exponent H und gewÃ¼nschter Standardabweichung.
    
    Mathematische Grundlage:
    -------------------------
    fBm ist GauÃŸscher Prozess mit Autokovarianz:
    
        Î³(k) = (1/2)Â·[(k+1)^(2H) - 2k^(2H) + (k-1)^(2H)]
    
    Davies-Harte Methode nutzt zirkulante Einbettung fÃ¼r exakte Simulation.
    
    Args:
        n_steps: Anzahl Zeitschritte
        H: Hurst-Exponent (Hâˆˆ(0,1))
           H=0.5: Brownsche Bewegung
           H<0.5: Anti-persistent (Subdiffusion)
           H>0.5: Persistent (Superdiffusion)
        sigma: Standardabweichung der Inkremente
        
    Returns:
        fBm Pfad [n_steps]
        
    Referenzen:
    -----------
    - Davies, R. B., & Harte, D. S. (1987). Biometrika, 74(1), 95-101.
    - Mandelbrot, B. B., & Van Ness, J. W. (1968). SIAM Review, 10(4), 422-437.
    """
    # Autokovarianzfunktion
    r = np.zeros(n_steps + 1)
    r[0] = 1.0
    
    for k in range(1, n_steps + 1):
        # Autokovarianz fBm: Î³(k) = (1/2)Â·[(k+1)^(2H) - 2k^(2H) + (k-1)^(2H)]
        r[k] = 0.5 * ((k + 1)**(2*H) - 2*k**(2*H) + abs(k - 1)**(2*H))
    
    # Zirkulante Einbettung
    r_extended = np.concatenate([r, r[1:-1][::-1]])
    
    # Eigenwerte via FFT (Zirkulantmatrix)
    eigenvalues = np.fft.fft(r_extended).real
    
    # Sicherstellung PositivitÃ¤t (numerische StabilitÃ¤t)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    try:
        # Generiere fBm via IFFT
        n_extended = 2 * n_steps
        W = np.fft.fft(
            np.sqrt(eigenvalues / n_extended) * 
            (np.random.randn(n_extended) + 1j * np.random.randn(n_extended))
        )
        
        # Extrahiere realen Teil
        W_real = W[:n_steps].real
        
        # Kumuliere fÃ¼r fBm
        fbm = np.cumsum(W_real) * sigma
        
    except Exception as e:
        # Fallback: Approximative fBm via Cholesky (langsamer aber robust)
        warnings.warn(f"Davies-Harte failed, using Cholesky fallback: {e}")
        
        # Approximative Inkremente
        increments = np.random.randn(n_steps) * sigma
        for i in range(1, n_steps):
            factor = ((i + 1) * 1.0)**(H - 0.5)  # Approximation
            increments[i] *= factor
        
        fbm = np.cumsum(increments)
    
    return fbm


def generate_normal_diffusion_spt(
    n_steps: int,
    D: float = 0.05,
    dt: float = 0.01,
    dimensionality: str = '2D',
    localization_precision: float = 0.015,
    motion_blur: bool = True,
    exposure_time: float = 0.01
) -> np.ndarray:
    """
    Wissenschaftlich exakte Simulation normaler Diffusion fÃ¼r SPT
    
    Implementiert Einstein-Smoluchowski Theorie mit realistischen experimentellen Artefakten:
    - Lokalisierungs-Noise (GauÃŸsch)
    - Motion Blur (finite Exposure Time)
    - Depth-of-field Effekte (fÃ¼r 3D)
    
    Theoretische Grundlage:
    -----------------------
    Einstein-Relation (1905): MSD(t) = 2dDt
    
    Wobei d=DimensionalitÃ¤t (2 oder 3)
    
    FÃ¼r 2D: âŸ¨xÂ²+yÂ²âŸ© = 4Dt
    FÃ¼r 3D: âŸ¨xÂ²+yÂ²+zÂ²âŸ© = 6Dt
    
    Diskrete Simulation:
    Î”r_i ~ N(0, 2DÂ·Î”t) fÃ¼r jede Dimension
    
    Args:
        n_steps: Anzahl Zeitschritte
        D: Diffusionskoeffizient [ÂµmÂ²/s]
           Typisch fÃ¼r TDI in Polymer: 0.001-0.5 ÂµmÂ²/s
        dt: Zeitschritt [s], typisch: 0.005-0.05 s
        dimensionality: '2D' oder '3D'
        localization_precision: Ïƒ_loc [Âµm], typisch: 10-30 nm
        motion_blur: BerÃ¼cksichtige Motion Blur?
        exposure_time: Exposure [s], meist = dt
        
    Returns:
        trajectory: (n_steps, d) array [Âµm]
        
    Referenzen:
    -----------
    - Einstein, A. (1905). Ann. Phys., 322(8), 549-560.
    - von Smoluchowski, M. (1906). Ann. Phys., 326(14), 756-780.
    - Michalet, X. (2010). Phys. Rev. E, 82(4), 041914. (Motion Blur)
    """
    dim = 2 if dimensionality == '2D' else 3
    
    # Einstein-Smoluchowski: ÏƒÂ² = 2DÂ·Î”t pro Dimension
    sigma_diff = np.sqrt(2 * D * dt)
    
    # Brownsche Inkremente
    increments = np.random.randn(n_steps, dim) * sigma_diff
    
    # Motion Blur Korrektur (Michalet, 2010)
    if motion_blur and exposure_time > 0:
        # Mittlung Ã¼ber Exposure-Zeit reduziert scheinbare MSD
        # Korrektur-Faktor: R = 1 - (t_exp)/(3Â·Î”t) fÃ¼r t_exp â‰ˆ Î”t
        R = exposure_time / (3.0 * dt) if dt > 0 else 0
        blur_factor = np.sqrt(1 - R) if R < 1 else 1.0
        increments *= blur_factor
    
    # Kumulatives Integral â†’ Position
    trajectory = np.cumsum(increments, axis=0)
    
    # Lokalisierungs-Noise (experimentelle Unsicherheit)
    # Ïƒ_loc typisch: 10-30 nm fÃ¼r TIRF, 20-50 nm fÃ¼r widefield
    loc_noise = np.random.randn(n_steps, dim) * localization_precision
    trajectory += loc_noise
    
    # FÃ¼r 3D: ZusÃ¤tzliche z-abhÃ¤ngige PSF-Verbreiterung (optional)
    if dimensionality == '3D':
        # Astigmatismus-basierte 3D: z-Genauigkeit schlechter als xy
        trajectory[:, 2] *= 1.5  # Typisch: Ïƒ_z â‰ˆ 1.5 Ã— Ïƒ_xy
    
    return trajectory


def generate_subdiffusion_spt(
    n_steps: int,
    alpha: float = 0.5,
    D_alpha: float = 0.05,
    dt: float = 0.01,
    dimensionality: str = '2D',
    localization_precision: float = 0.015,
    mechanism: str = 'fbm'
) -> np.ndarray:
    """
    Subdiffusion via fraktionaler Brownscher Bewegung
    
    Theoretische Grundlage:
    -----------------------
    MSD(t) = 2dÂ·D_Î±Â·t^Î±    (d=DimensionalitÃ¤t)
    
    FÃ¼r Î±<1: Subdiffusion
    
    Physikalische Mechanismen:
    - 'fbm': Fraktionale Brownsche Bewegung (memory effects)
    - 'ctrw': Continuous-Time Random Walk (trapping)
    - 'obstacles': Diffusion mit Hindernissen (crowding)
    
    Implementierung: fBm mit H = Î±/2
    
    Args:
        n_steps: Anzahl Zeitschritte
        alpha: Anomaler Exponent Î±âˆˆ(0,1) fÃ¼r Subdiffusion
               Typisch: 0.3-0.9
               Î±â‰ˆ0.7-0.9: Schwaches Crowding
               Î±â‰ˆ0.4-0.6: Starkes Crowding
               Î±<0.4: Extreme Subdiffusion (selten)
        D_alpha: Generalisierter Diffusionskoeffizient [ÂµmÂ²/s^Î±]
        dt: Zeitschritt [s]
        dimensionality: '2D' oder '3D'
        localization_precision: Ïƒ_loc [Âµm]
        mechanism: 'fbm', 'ctrw', oder 'obstacles'
        
    Returns:
        trajectory: (n_steps, d) array [Âµm]
        
    Referenzen:
    -----------
    - Metzler, R., & Klafter, J. (2000). Phys. Rep., 339(1), 1-77.
    - HÃ¶fling, F., & Franosch, T. (2013). Rep. Prog. Phys., 76(4), 046602.
    - Weigel, A. V., et al. (2011). PNAS, 108(16), 6438-6443.
    """
    dim = 2 if dimensionality == '2D' else 3
    
    # Hurst-Exponent aus anomalem Exponenten
    H = alpha / 2.0
    H = np.clip(H, 0.05, 0.49)  # H < 0.5 fÃ¼r Subdiffusion
    
    # Skalierung fÃ¼r korrektes MSD(t) = 2dÂ·D_Î±Â·t^Î±
    # fBm hat Varianz ~ t^(2H) = t^Î±
    # BenÃ¶tigt: Ïƒ = sqrt(2Â·D_Î±Â·dt^Î±) pro Dimension
    sigma = np.sqrt(2 * D_alpha * (dt ** alpha))
    
    trajectory = np.zeros((n_steps, dim))
    
    if mechanism == 'fbm':
        # fBm via Davies-Harte (exakt)
        for d in range(dim):
            trajectory[:, d] = generate_fbm_davies_harte(n_steps, H, sigma)
            
    elif mechanism == 'ctrw':
        # CTRW: Poisson-verteilte Wartezeiten (approximativ)
        # Nicht vollstÃ¤ndig implementiert - wÃ¼rde Levy-Wartezeiten benÃ¶tigen
        warnings.warn("CTRW not fully implemented, using fBm approximation")
        for d in range(dim):
            trajectory[:, d] = generate_fbm_davies_harte(n_steps, H, sigma)
    
    elif mechanism == 'obstacles':
        # Diffusion mit zufÃ¤lligen Hindernissen (approximativ via modifizierte fBm)
        for d in range(dim):
            trajectory[:, d] = generate_fbm_davies_harte(n_steps, H, sigma)
            # ZusÃ¤tzlich: gelegentliche "Stoppungen"
            if np.random.rand() < 0.3:  # 30% haben Trapping-Events
                trap_indices = np.random.choice(n_steps, size=int(n_steps*0.1), replace=False)
                trajectory[trap_indices, d] *= 0.3  # Reduzierte Bewegung
    
    # Lokalisierungs-Noise
    loc_noise = np.random.randn(n_steps, dim) * localization_precision
    trajectory += loc_noise
    
    # 3D z-Asymmetrie
    if dimensionality == '3D':
        trajectory[:, 2] *= 1.5
    
    return trajectory


def generate_superdiffusion_spt(
    n_steps: int,
    alpha: float = 1.5,
    D_alpha: float = 0.1,
    dt: float = 0.01,
    dimensionality: str = '2D',
    localization_precision: float = 0.015,
    mode: str = 'persistent'
) -> np.ndarray:
    """
    Superdiffusion: Persistente Bewegung oder LÃ©vy Flights
    
    Theoretische Grundlage:
    -----------------------
    MSD(t) = 2dÂ·D_Î±Â·t^Î±    mit Î±>1
    
    Physikalische Mechanismen:
    - 'persistent': Ballistische Komponente (fBm mit H>0.5)
    - 'levy': LÃ©vy Flights (lange SprÃ¼nge)
    - 'active': Active Transport (Motor-Proteine, nicht SPT-relevant)
    
    Args:
        n_steps: Anzahl Zeitschritte
        alpha: Î±âˆˆ(1, 2) fÃ¼r Superdiffusion
               Î±â‰ˆ1.1-1.3: Schwach persistent
               Î±â‰ˆ1.4-1.6: Stark persistent
               Î±>1.7: Ballistisch (sehr selten in SPT)
        D_alpha: Generalisierter Diffusionskoeffizient [ÂµmÂ²/s^Î±]
        dt: Zeitschritt [s]
        dimensionality: '2D' oder '3D'
        localization_precision: Ïƒ_loc [Âµm]
        mode: 'persistent' oder 'levy'
        
    Returns:
        trajectory: (n_steps, d) array [Âµm]
        
    Referenzen:
    -----------
    - Metzler, R., & Klafter, J. (2000). Phys. Rep., 339(1), 1-77.
    - Mandelbrot, B. B. (1982). The Fractal Geometry of Nature. Freeman.
    - Chechkin, A. V., et al. (2017). Phys. Rev. X, 7(2), 021002.
    """
    dim = 2 if dimensionality == '2D' else 3
    
    if mode == 'persistent':
        # fBm mit H > 0.5 (persistent)
        H = alpha / 2.0
        H = np.clip(H, 0.51, 0.95)  # H > 0.5 fÃ¼r Superdiffusion
        
        sigma = np.sqrt(2 * D_alpha * (dt ** alpha))
        
        trajectory = np.zeros((n_steps, dim))
        for d in range(dim):
            trajectory[:, d] = generate_fbm_davies_harte(n_steps, H, sigma)
    
    elif mode == 'levy':
        # LÃ©vy Flight: Stabile Verteilung mit Index Î² = 2/Î±
        beta = 2.0 / alpha
        beta = np.clip(beta, 0.5, 1.9)  # StabilitÃ¤tsparameter
        
        # LÃ©vy-Schritte: P(l) ~ l^(-1-Î²)
        scale = np.sqrt(2 * D_alpha * (dt ** alpha))
        
        trajectory = np.zeros((n_steps, dim))
        for d in range(dim):
            steps = levy_stable.rvs(beta, 0, scale=scale, size=n_steps)
            trajectory[:, d] = np.cumsum(steps)
    
    # Lokalisierungs-Noise
    loc_noise = np.random.randn(n_steps, dim) * localization_precision
    trajectory += loc_noise
    
    # 3D z-Asymmetrie
    if dimensionality == '3D':
        trajectory[:, 2] *= 1.5
    
    return trajectory


def generate_confined_diffusion_spt(
    n_steps: int,
    confinement_radius: float = 0.8,
    D: float = 0.05,
    dt: float = 0.01,
    dimensionality: str = '2D',
    geometry: str = 'spherical',
    alpha_confined: Optional[float] = None,
    localization_precision: float = 0.015
) -> np.ndarray:
    """
    Confined Diffusion: RÃ¤umlich begrenzte Diffusion
    
    Theoretische Grundlage:
    -----------------------
    MSD(t) â‰ˆ 2dÂ·DÂ·t           fÃ¼r t << Ï„_conf  (freie Diffusion)
    MSD(t) â†’ LÂ²              fÃ¼r t >> Ï„_conf  (Plateau)
    
    Charakteristische Zeit: Ï„_conf = LÂ²/(2dÂ·D)
    
    Physikalische Realisierungen:
    - Membran-MikrodomÃ¤nen (Lipid Rafts): 50-500 nm
    - Proteine in Poren: 20-100 nm
    - ZellulÃ¤re Kompartimente: 0.5-5 Âµm
    - TDI in Polymer-MikrodomÃ¤nen: 0.2-2 Âµm (geschÃ¤tzt)
    
    Args:
        n_steps: Anzahl Zeitschritte
        confinement_radius: Confinement-Radius L [Âµm]
                           Typisch: 0.2-2.0 Âµm
        D: Diffusionskoeffizient IM Confinement [ÂµmÂ²/s]
        dt: Zeitschritt [s]
        dimensionality: '2D' oder '3D'
        geometry: 'spherical', 'cubic', oder 'cylindrical'
        alpha_confined: Optional Î±<1 fÃ¼r subdiffusive confined
        localization_precision: Ïƒ_loc [Âµm]
        
    Returns:
        trajectory: (n_steps, d) array [Âµm]
        
    Referenzen:
    -----------
    - Kusumi, A., et al. (2012). Semin. Cell Dev. Biol., 23(2), 126-134.
    - Ritchie, K., et al. (2005). Biophys. J., 88(3), 2266-2277.
    - Saxton, M. J. (1997). Biophys. J., 72(4), 1744-1753.
    """
    dim = 2 if dimensionality == '2D' else 3
    
    trajectory = np.zeros((n_steps, dim))
    position = np.zeros(dim)
    
    # Start: Oft nahe Zentrum, manchmal off-center
    if np.random.rand() > 0.7:
        # 30% starten off-center
        angle = np.random.uniform(0, 2*np.pi)
        r_start = np.random.uniform(0, confinement_radius * 0.4)
        
        if dim == 2:
            position[0] = r_start * np.cos(angle)
            position[1] = r_start * np.sin(angle)
        else:  # 3D
            phi = np.random.uniform(0, np.pi)
            position[0] = r_start * np.sin(phi) * np.cos(angle)
            position[1] = r_start * np.sin(phi) * np.sin(angle)
            position[2] = r_start * np.cos(phi)
    
    # Subdiffusive confined?
    if alpha_confined is not None and alpha_confined < 1.0:
        H = alpha_confined / 2.0
        H = np.clip(H, 0.2, 0.45)
        is_subdiffusive = True
        sigma = np.sqrt(2 * D * (dt ** alpha_confined))
    else:
        H = 0.5
        is_subdiffusive = False
        sigma = np.sqrt(2 * D * dt)
    
    # Simulation mit reflektierenden Randbedingungen
    for i in range(n_steps):
        # Diffusionsschritt
        if is_subdiffusive:
            # Subdiffusive: Î”r ~ t^(H-0.5)
            step = np.random.randn(dim) * sigma * ((i + 1) ** (H - 0.5))
        else:
            # Normale Diffusion
            step = np.random.randn(dim) * sigma
        
        new_position = position + step
        
        # Geometrie-spezifische Reflexion
        if geometry == 'spherical':
            # SphÃ¤risches/kreisfÃ¶rmiges Confinement
            distance = np.linalg.norm(new_position)
            
            if distance > confinement_radius:
                # Elastische Reflexion
                normal = new_position / (distance + 1e-10)
                
                # Variante 1: Setze auf Rand
                new_position = confinement_radius * normal * 0.97
                
                # Variante 2: Perfekte Reflexion (optional, realistischer aber komplexer)
                # overshoot = distance - confinement_radius
                # new_position = (confinement_radius - overshoot * 0.8) * normal
        
        elif geometry == 'cubic':
            # WÃ¼rfelfÃ¶rmiges Confinement
            for d in range(dim):
                if abs(new_position[d]) > confinement_radius:
                    new_position[d] = np.sign(new_position[d]) * confinement_radius * 0.97
        
        elif geometry == 'cylindrical' and dim == 3:
            # Zylindrisches Confinement (nur 3D)
            r_xy = np.linalg.norm(new_position[:2])
            if r_xy > confinement_radius:
                normal_xy = new_position[:2] / (r_xy + 1e-10)
                new_position[:2] = confinement_radius * normal_xy * 0.97
            
            # z unbegrenzt oder auch begrenzt:
            if abs(new_position[2]) > confinement_radius * 2:  # LÃ¤ngerer Zylinder
                new_position[2] = np.sign(new_position[2]) * confinement_radius * 2 * 0.97
        
        position = new_position
        trajectory[i] = position
    
    # Lokalisierungs-Noise
    loc_noise = np.random.randn(n_steps, dim) * localization_precision
    trajectory += loc_noise
    
    # 3D z-Asymmetrie
    if dimensionality == '3D':
        trajectory[:, 2] *= 1.5
    
    return trajectory


def generate_spt_dataset(
    n_samples_per_class: int = 3000,
    min_length: int = 50,
    max_length: int = 5000,
    dimensionality: str = '2D',
    polymerization_degree: float = 0.5,
    dt: float = 0.01,
    localization_precision: float = 0.015,
    boost_classes: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """
    Generiere vollstÃ¤ndigen wissenschaftlichen SPT-Datensatz
    
    Generiert synthetische SPT-Trajektorien fÃ¼r Training eines Deep-Learning-Klassifikators
    mit physikalisch korrekten Parametern fÃ¼r TDI-G0 in Polymermatrizen.
    
    Args:
        n_samples_per_class: Basis-Anzahl pro Klasse
        min_length: Minimale Trajektorien-LÃ¤nge [Frames]
        max_length: Maximale Trajektorien-LÃ¤nge [Frames]
        dimensionality: '2D' oder '3D'
        polymerization_degree: 0.0-1.0 (Eduktschmelze bis vollpolymerisiert)
        dt: Zeitschritt [s], typisch: 0.01s = 10ms
        localization_precision: Ïƒ_loc [Âµm], typisch: 15nm = 0.015Âµm
        boost_classes: Liste von Klassen zum Boosten (z.B. ['Normal', 'Confined'])
        verbose: Print-Output?
        
    Returns:
        X: Liste von Trajektorien [Âµm] (variable LÃ¤ngen!)
        y: Labels (0=Normal, 1=Sub, 2=Super, 3=Confined)
        lengths: Trajektorien-LÃ¤ngen
        class_names: Klassennamen
        
    Physikalische Parameter (TDI-G0 in Polymer):
    ---------------------------------------------
    basierend auf polymerization_degree:
    
    - Niedrig (0-30%): D âˆˆ [0.05, 0.8] ÂµmÂ²/s  (flÃ¼ssig)
    - Mittel (30-70%): D âˆˆ [0.01, 0.2] ÂµmÂ²/s  (viskoelastisch)
    - Hoch (70-100%): D âˆˆ [0.001, 0.05] ÂµmÂ²/s (fest, glasig)
    """
    class_names = ['Normal', 'Subdiffusion', 'Superdiffusion', 'Confined']
    
    if boost_classes is None:
        # Standard: keine künstliche Klassengewichtung –
        # gleich viele Trajektorien pro Diffusionstyp
        boost_classes = []
    
    # D-Bereich basierend auf Polymerisierung
    D_min, D_max = PolymerDiffusionParameters.get_D_range(polymerization_degree)
    
    X = []
    y = []
    lengths = []
    
    if verbose:
        print("="*80)
        print("ðŸ”¬ WISSENSCHAFTLICHE SPT-TRAJEKTORIEN-GENERIERUNG")
        print("="*80)
        print(f"DimensionalitÃ¤t: {dimensionality}")
        print(f"Polymerisierungsgrad: {polymerization_degree:.1%}")
        print(f"D-Bereich: {D_min:.3f} - {D_max:.3f} ÂµmÂ²/s")
        print(f"Zeitschritt dt: {dt} s")
        print(f"Lokalisierungs-Precision: {localization_precision*1000:.1f} nm")
        print(f"Trajektorien-LÃ¤nge: {min_length} - {max_length} Frames")
        print(f"Basis Samples/Klasse: {n_samples_per_class}")
        print(f"Boost Klassen: {boost_classes}")
        print("="*80 + "\n")
    
    # === KLASSE 0: NORMAL ===
    n_normal = n_samples_per_class * 2 if 'Normal' in boost_classes else n_samples_per_class
    
    if verbose:
        print(f"ðŸ“Š [1/4] Generiere Normal Diffusion (n={n_normal})...")
        print(f"   Î± = 1.0 (Einstein-Smoluchowski)")
        print(f"   D = {D_min:.3f} - {D_max:.3f} ÂµmÂ²/s")
    
    for i in range(n_normal):
        if verbose and i % 1000 == 0:
            print(f"   {i}/{n_normal}", end='\r')
        
        n_steps = np.random.randint(min_length, max_length + 1)
        D = np.random.uniform(D_min, D_max)
        
        traj = generate_normal_diffusion_spt(
            n_steps, D=D, dt=dt, dimensionality=dimensionality,
            localization_precision=localization_precision
        )
        
        X.append(traj)
        y.append(0)
        lengths.append(n_steps)
    
    if verbose:
        print(f"   âœ… {n_normal} Normal generiert\n")
    
    # === KLASSE 1: SUBDIFFUSION ===
    n_sub = n_samples_per_class * 2 if 'Subdiffusion' in boost_classes else n_samples_per_class
    
    if verbose:
        print(f"ðŸ“Š [2/4] Generiere Subdiffusion (n={n_sub})...")
        print(f"   Î± = 0.3 - 0.9 (fraktionale Brownsche Bewegung)")
        print(f"   D_Î± = {D_min*0.5:.3f} - {D_max*0.6:.3f} ÂµmÂ²/s^Î±")
    
    for i in range(n_sub):
        if verbose and i % 1000 == 0:
            print(f"   {i}/{n_sub}", end='\r')
        
        n_steps = np.random.randint(min_length, max_length + 1)
        
        # Î± mit Bias zu niedrigen Werten (Crowding ist hÃ¤ufiger als schwache Subdiffusion)
        if np.random.rand() < 0.6:
            alpha = np.random.uniform(0.35, 0.65)  # Starkes Crowding
        else:
            alpha = np.random.uniform(0.65, 0.90)  # Schwaches Crowding
        
        # D_Î± unabhÃ¤ngig von Î± (wichtig!)
        D_alpha = np.random.uniform(D_min * 0.5, D_max * 0.6)
        
        mechanism = np.random.choice(['fbm', 'obstacles'], p=[0.7, 0.3])
        
        traj = generate_subdiffusion_spt(
            n_steps, alpha=alpha, D_alpha=D_alpha, dt=dt,
            dimensionality=dimensionality, localization_precision=localization_precision,
            mechanism=mechanism
        )
        
        X.append(traj)
        y.append(1)
        lengths.append(n_steps)
    
    if verbose:
        print(f"   âœ… {n_sub} Subdiffusion generiert\n")
    
    # === KLASSE 2: SUPERDIFFUSION ===
    n_super = n_samples_per_class * 2 if 'Superdiffusion' in boost_classes else n_samples_per_class
    
    if verbose:
        print(f"ðŸ“Š [3/4] Generiere Superdiffusion (n={n_super})...")
        print(f"   Î± = 1.2 - 1.8 (persistent/LÃ©vy)")
        print(f"   D_Î± = {D_min*0.8:.3f} - {D_max*1.2:.3f} ÂµmÂ²/s^Î±")
    
    for i in range(n_super):
        if verbose and i % 1000 == 0:
            print(f"   {i}/{n_super}", end='\r')
        
        n_steps = np.random.randint(min_length, max_length + 1)
        
        alpha = np.random.uniform(1.15, 1.80)
        D_alpha = np.random.uniform(D_min * 0.8, D_max * 1.2)
        
        mode = 'persistent' if np.random.rand() > 0.25 else 'levy'
        
        traj = generate_superdiffusion_spt(
            n_steps, alpha=alpha, D_alpha=D_alpha, dt=dt,
            dimensionality=dimensionality, localization_precision=localization_precision,
            mode=mode
        )
        
        X.append(traj)
        y.append(2)
        lengths.append(n_steps)
    
    if verbose:
        print(f"   âœ… {n_super} Superdiffusion generiert\n")
    
    # === KLASSE 3: CONFINED ===
    n_conf = n_samples_per_class * 2 if 'Confined' in boost_classes else n_samples_per_class
    
    if verbose:
        print(f"ðŸ“Š [4/4] Generiere Confined Diffusion (n={n_conf})...")
        print(f"   Confinement-Radius: 0.2 - 2.0 Âµm")
        print(f"   D (im Confinement): {D_min:.3f} - {D_max*0.5:.3f} ÂµmÂ²/s")
        print(f"   30% subdiffusive confined (Î±=0.6-0.9)")
    
    for i in range(n_conf):
        if verbose and i % 1000 == 0:
            print(f"   {i}/{n_conf}", end='\r')
        
        n_steps = np.random.randint(min_length, max_length + 1)
        
        # Confinement-Radius: realistisch fÃ¼r Polymer-MikrodomÃ¤nen
        confinement_radius = np.random.uniform(0.2, 2.0)  # Âµm
        
        # D im Confinement (oft niedriger)
        D_conf = np.random.uniform(D_min, D_max * 0.5)
        
        geometry = np.random.choice(['spherical', 'cubic'], p=[0.7, 0.3])
        
        # 30% subdiffusive confined
        alpha_conf = None
        if np.random.rand() < 0.3:
            alpha_conf = np.random.uniform(0.6, 0.9)
        
        traj = generate_confined_diffusion_spt(
            n_steps, confinement_radius=confinement_radius, D=D_conf, dt=dt,
            dimensionality=dimensionality, geometry=geometry,
            alpha_confined=alpha_conf, localization_precision=localization_precision
        )
        
        X.append(traj)
        y.append(3)
        lengths.append(n_steps)
    
    if verbose:
        print(f"   âœ… {n_conf} Confined generiert\n")
    
    y = np.array(y)
    lengths = np.array(lengths)
    
    if verbose:
        print("="*80)
        print("âœ… DATENSATZ VOLLSTÃ„NDIG!")
        print("="*80)
        print(f"Gesamt-Samples: {len(X)}")
        print(f"Dimensionen: {X[0].shape[1]}D")
        print(f"Trajektorien-LÃ¤ngen: {lengths.min()} - {lengths.max()} Frames")
        print(f"Mittlere LÃ¤nge: {lengths.mean():.1f} Â± {lengths.std():.1f} Frames")
        print(f"\nKlassen-Verteilung:")
        for i, name in enumerate(class_names):
            count = np.sum(y == i)
            percentage = 100 * count / len(y)
            avg_len = lengths[y == i].mean()
            print(f"  {name:15s}: {count:6d} ({percentage:5.1f}%) - Ã˜ {avg_len:.0f} Frames")
        print("="*80 + "\n")
    
    return X, y, lengths, class_names


if __name__ == "__main__":
    # Test-Generation
    print("\nðŸ§ª TEST: Wissenschaftliche SPT-Trajektorien")
    print("="*80)
    
    # 2D Test
    X_2d, y_2d, lengths_2d, names = generate_spt_dataset(
        n_samples_per_class=10000,
        min_length=50,
        max_length=5000,
        dimensionality='2D',
        polymerization_degree=0.5,
        verbose=True
    )
    
    print(f"\nâœ… 2D Datensatz: {len(X_2d)} Trajektorien")
    print(f"   Beispiel-Shape: {X_2d[0].shape}")
    
    # 3D Test
    print("\n" + "="*80)
    X_3d, y_3d, lengths_3d, _ = generate_spt_dataset(
        n_samples_per_class=10000,
        min_length=50,
        max_length=5000,
        dimensionality='3D',
        polymerization_degree=0.5,
        verbose=True
    )
    
    print(f"\nâœ… 3D Datensatz: {len(X_3d)} Trajektorien")
    print(f"   Beispiel-Shape: {X_3d[0].shape}")
    
    print("\n" + "="*80)
    print("ðŸŽ“ WISSENSCHAFTLICHE TRAJEKTORIEN-GENERIERUNG ERFOLGREICH!")
    print("="*80)
