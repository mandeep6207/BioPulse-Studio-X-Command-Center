import numpy as np
import scipy.signal
from scipy.stats import kurtosis, skew
from typing import Dict, Any, Tuple, List
from biopulse_backend.utils.math_utils import calculate_entropy, calculate_snr_and_drift

def calculate_beat_consistency(signal: np.ndarray, fs: float, peaks: np.ndarray) -> float:
    """
    Segments the signal into beats, interpolates them to 100 points,
    and returns the mean pairwise correlation coefficient of all beats.
    """
    if len(peaks) < 4:
        return 0.5 # Default neutral value if not enough beats
        
    beats = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        beat = signal[start_idx:end_idx]
        if len(beat) > 5:
            # Interpolate to 100 points
            x = np.linspace(0, 1, len(beat))
            x_new = np.linspace(0, 1, 100)
            beat_resampled = np.interp(x_new, x, beat)
            # Subtract mean and normalize
            beat_resampled = beat_resampled - np.mean(beat_resampled)
            std = np.std(beat_resampled)
            if std > 1e-9:
                beat_resampled = beat_resampled / std
                beats.append(beat_resampled)
                
    if len(beats) < 3:
        return 0.5
        
    # Calculate pairwise correlation coefficients
    beats_arr = np.array(beats) # shape (M, 100)
    corr_matrix = np.corrcoef(beats_arr) # shape (M, M)
    
    # Get upper triangle elements excluding diagonal
    indices = np.triu_indices_from(corr_matrix, k=1)
    corrs = corr_matrix[indices]
    
    # Filter out NaNs if any
    corrs = corrs[~np.isnan(corrs)]
    
    if len(corrs) == 0:
        return 0.5
        
    return float(np.mean(corrs))

def assess_signal_quality(
    raw_signal: np.ndarray, 
    interpolated_signal: np.ndarray, 
    fs: float
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Computes multiple signal metrics and calculates a unified quality score between 0 and 100,
    mapped to: Excellent (95-100), Very Good (85-94), Good (70-84), Acceptable (50-69), Poor (<50).
    """
    # 1. Missing data % (count differences between raw and interpolated)
    # Or count exact NaNs in raw_signal
    total_len = len(raw_signal)
    nans = np.isnan(raw_signal) | np.isinf(raw_signal)
    missing_pct = float(np.sum(nans) / total_len * 100.0)
    
    # Use interpolated signal for subsequent DSP calculations
    # Apply robust outlier clipping to avoid startup/transient artifact poisoning
    sig = interpolated_signal.copy()
    q25, q75 = np.percentile(sig, [25, 75])
    iqr = q75 - q25
    med = np.median(sig)
    if iqr > 1e-9:
        lower = med - 3.5 * iqr
        upper = med + 3.5 * iqr
        sig = np.clip(sig, lower, upper)
    
    # 2. Flat region check
    diffs = np.diff(sig)
    flat_samples = np.sum(np.abs(diffs) < 1e-7)
    flat_pct = float(flat_samples / len(diffs) * 100.0)
    
    # 3. Basic statistics
    sig_var = float(np.var(sig))
    sig_std = np.std(sig)
    sig_kurt = float(kurtosis(sig))
    sig_skew = float(skew(sig))
    
    # 4. Entropy
    ent = calculate_entropy(sig)
    
    # 5. SNR and baseline drift
    snr, drift_std = calculate_snr_and_drift(sig, fs)
    drift_ratio = float(drift_std / sig_std) if sig_std > 1e-9 else 0.0
    
    # 6. Motion artifact estimate
    # Detect sudden large derivative spikes (outliers > 5 * std)
    diff_std = np.std(diffs)
    if diff_std > 1e-9:
        motion_spikes = np.sum(np.abs(diffs) > 5.0 * diff_std)
        motion_pct = float(motion_spikes / len(diffs) * 100.0)
    else:
        motion_pct = 0.0
        
    # 7. Peak detection, stability, and beat consistency
    # Detrend for clean peak finding
    sig_detrend = scipy.signal.detrend(sig)
    dist = max(5, int(fs * 0.4)) # ~150 bpm max
    peaks, props = scipy.signal.find_peaks(sig_detrend, distance=dist, prominence=0.1 * np.std(sig_detrend))
    
    peak_stability = 0.0
    consistency = 0.85 # default neutral consistent
    
    if len(peaks) > 1:
        proms = props["prominences"]
        mean_prom = np.mean(proms)
        if mean_prom > 1e-9:
            peak_stability = float(np.std(proms) / mean_prom)
        
        # Calculate beat consistency
        consistency = calculate_beat_consistency(sig_detrend, fs, peaks)
        
    # --- Scoring engine ---
    score = 100.0
    
    # Deductions
    # Missing data
    score -= missing_pct * 3.0
    
    # Flat regions
    if flat_pct > 1.0:
        score -= (flat_pct - 1.0) * 2.0
        
    # SNR
    if snr < 15.0:
        score -= (15.0 - snr) * 2.0
    if snr < 0.0:
        score -= 10.0 # additional penalty for extremely poor SNR
        
    # Baseline drift
    if drift_ratio > 0.1:
        score -= (drift_ratio - 0.1) * 40.0
        
    # Motion artifacts
    score -= motion_pct * 15.0
    
    # Peak instability
    score -= peak_stability * 25.0
    
    # Beat consistency
    if consistency < 0.85:
        score -= (0.85 - consistency) * 60.0
        
    # Clamp score
    score = float(np.clip(score, 0.0, 100.0))
    
    # Map to bands
    if score >= 95.0:
        band = "Excellent"
    elif score >= 85.0:
        band = "Very Good"
    elif score >= 70.0:
        band = "Good"
    elif score >= 50.0:
        band = "Acceptable"
    else:
        band = "Poor"
        
    metrics = {
        "snr_db": snr,
        "baseline_wander_std": drift_std,
        "drift_ratio": drift_ratio,
        "motion_artifact_pct": motion_pct,
        "entropy": ent,
        "variance": sig_var,
        "kurtosis": sig_kurt,
        "skewness": sig_skew,
        "flat_region_pct": flat_pct,
        "missing_data_pct": missing_pct,
        "peak_stability": peak_stability,
        "beat_consistency": consistency,
        "peak_count": len(peaks)
    }
    
    return score, band, metrics
