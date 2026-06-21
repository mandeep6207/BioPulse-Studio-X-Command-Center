import numpy as np
import scipy.signal
from typing import Tuple, Dict, Any
from biopulse_backend.core.morphology_guard import extract_morphology_properties
from biopulse_backend.utils.math_utils import calculate_derivatives, calculate_entropy

def calculate_nonlinear_hrv(peaks: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """
    Computes Poincare plot features SD1, SD2, and SD1/SD2 ratio from peak indices.
    """
    if len(peaks) < 3:
        return 0.0, 0.0, 1.0
        
    rr_intervals = np.diff(peaks) / fs
    x = rr_intervals[:-1]
    y = rr_intervals[1:]
    
    # SD1: short-term HRV (parasympathetic)
    sd1 = float(np.sqrt(0.5 * np.var(x - y)))
    # SD2: long-term HRV
    sd2 = float(np.sqrt(0.5 * np.var(x + y)))
    # SD1/SD2 ratio
    ratio = sd1 / sd2 if sd2 > 1e-9 else 1.0
    
    return sd1, sd2, ratio

def validate_features(
    signal: np.ndarray, 
    fs: float, 
    fs_status: str
) -> Tuple[float, str, Dict[str, Any], Dict[str, Any]]:
    """
    Validates signal morphology, derivatives, entropy, energy, and nonlinear features
    using physiological plausibility checks.
    Enforces Readiness constraints if fs_status is 'FS_UNRESOLVED'.
    """
    # 1. Calculate derivatives & energy
    vpg, apg = calculate_derivatives(signal, fs)
    sig_detrend = scipy.signal.detrend(signal)
    energy = float(np.sum(sig_detrend ** 2) / len(signal))
    
    # 2. Extract morphology properties
    morph = extract_morphology_properties(signal, fs)
    
    # 3. Calculate entropy
    ent = calculate_entropy(signal)
    
    # 4. Calculate nonlinear features
    dist = max(5, int(fs * 0.4))
    peaks, _ = scipy.signal.find_peaks(sig_detrend, distance=dist, prominence=0.1 * np.std(sig_detrend))
    sd1, sd2, sd_ratio = calculate_nonlinear_hrv(peaks, fs)
    
    # 5. Physiological plausibility checks
    checks = {}
    
    # Check 1: Heart rate range [40, 160] bpm
    checks["hr_plausible"] = (40.0 <= morph["hr"] <= 160.0)
    
    # Check 2: Mean RR interval range [0.375, 1.5] seconds
    checks["rr_plausible"] = (0.375 <= morph["rr"] <= 1.5)
    
    # Check 3: Pulse width [0.10, 0.60] seconds
    checks["pulse_width_plausible"] = (0.10 <= morph["pulse_width"] <= 0.60)
    
    # Check 4: Rise time [0.05, 0.45] seconds
    checks["rise_time_plausible"] = (0.05 <= morph["rise_time"] <= 0.45)
    
    # Check 5: Decay time [0.10, 1.20] seconds
    checks["decay_time_plausible"] = (0.10 <= morph["decay_time"] <= 1.20)
    
    # Check 6: Amplitude positive and non-flat
    checks["amplitude_positive"] = (morph["amplitude"] > 1e-6)
    
    # Check 7: Entropy range [1.5, 7.5] bits
    checks["entropy_plausible"] = (1.5 <= ent <= 7.5)
    
    # Check 8: SD1 physiological limit [0.001, 0.200] seconds
    checks["sd1_plausible"] = (0.001 <= sd1 <= 0.200)
    
    # Check 9: SD2 physiological limit [0.005, 0.400] seconds
    checks["sd2_plausible"] = (0.005 <= sd2 <= 0.400)
    
    # Check 10: VPG and APG peak-to-peak range
    # Ensure they are active and not flat
    vpg_range = float(np.max(vpg) - np.min(vpg))
    apg_range = float(np.max(apg) - np.min(apg))
    checks["derivatives_active"] = (vpg_range > 1e-6 and apg_range > 1e-6)
    
    # Count passed checks
    total_checks = len(checks)
    passed_checks = sum(1 for v in checks.values() if v)
    
    # Raw readiness score
    readiness_score = (passed_checks / total_checks) * 100.0
    
    # Categorization based on score
    if readiness_score >= 90.0:
        category = "A"
    elif readiness_score >= 75.0:
        category = "B"
    elif readiness_score >= 50.0:
        category = "C"
    else:
        category = "D"
        
    # Enforce Stage 3 cap if FS is unresolved
    if fs_status == "FS_UNRESOLVED":
        readiness_score = min(readiness_score, 85.0)
        if category == "A":
            category = "B"
            
    validated_features = {
        "heart_rate_bpm": morph["hr"],
        "rr_mean_sec": morph["rr"],
        "pulse_width_sec": morph["pulse_width"],
        "rise_time_sec": morph["rise_time"],
        "decay_time_sec": morph["decay_time"],
        "amplitude": morph["amplitude"],
        "entropy": ent,
        "energy": energy,
        "area": morph["area"],
        "sd1_sec": sd1,
        "sd2_sec": sd2,
        "sd_ratio": sd_ratio,
        "vpg_range": vpg_range,
        "apg_range": apg_range
    }
    
    validation_report = {
        "checks": checks,
        "passed_count": passed_checks,
        "total_count": total_checks,
        "fs_status": fs_status
    }
    
    return float(readiness_score), category, validated_features, validation_report
