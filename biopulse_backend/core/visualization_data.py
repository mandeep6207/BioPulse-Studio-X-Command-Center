import numpy as np
import scipy.signal
from typing import Dict, Any, List
from biopulse_backend.utils.math_utils import calculate_derivatives

def prepare_visualization_data(
    raw_signal: np.ndarray,
    preprocessed_signal: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    quality_metrics: Dict[str, Any],
    validated_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Shapes and prepares raw and processed signal vectors, peak locations, derivatives,
    trends, radar plot coordinates, and chunk-wise noise matrices for Plotly visualizations.
    No plotting packages are imported in this module.
    """
    n = len(raw_signal)
    time_array = np.arange(n) / fs
    
    # 1. Derivatives (VPG/APG)
    vpg, apg = calculate_derivatives(preprocessed_signal, fs)
    
    # 2. Trends (HR and RR series)
    rr_intervals = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    rr_times = peaks[1:] / fs if len(peaks) > 1 else np.array([])
    hr_trend = 60.0 / rr_intervals if len(rr_intervals) > 0 else np.array([])
    
    # 3. Quality Radar values (normalized 0 to 100)
    snr_db = quality_metrics.get("snr_db", 0.0)
    norm_snr = float(np.clip(snr_db / 30.0 * 100.0, 0.0, 100.0))
    
    drift_ratio = quality_metrics.get("drift_ratio", 0.0)
    norm_drift = float(np.clip((1.0 - drift_ratio) * 100.0, 0.0, 100.0))
    
    consistency = quality_metrics.get("beat_consistency", 0.5)
    norm_consistency = float(consistency * 100.0)
    
    flatness = quality_metrics.get("flat_region_pct", 0.0)
    norm_flatness = float(np.clip((100.0 - flatness), 0.0, 100.0))
    
    stability = quality_metrics.get("peak_stability", 0.0)
    norm_stability = float(np.clip((1.0 - stability) * 100.0, 0.0, 100.0))
    
    quality_radar = {
        "SNR (Power)": norm_snr,
        "Baseline Stability": norm_drift,
        "Beat Consistency": norm_consistency,
        "Sensor Connection": norm_flatness,
        "Amplitude Stability": norm_stability
    }
    
    # 4. Morphology Radar values (normalized relative to standard physiologic centers)
    # Target HR = 72, RR = 0.833, PulseWidth = 0.3, RiseTime = 0.2, Amplitude = 1.0
    hr = validated_features.get("heart_rate_bpm", 72.0)
    norm_hr = float(np.clip(100.0 - abs(hr - 72.0)/72.0 * 100.0, 0.0, 100.0))
    
    pw = validated_features.get("pulse_width_sec", 0.3)
    norm_pw = float(np.clip(100.0 - abs(pw - 0.3)/0.3 * 100.0, 0.0, 100.0))
    
    rt = validated_features.get("rise_time_sec", 0.2)
    norm_rt = float(np.clip(100.0 - abs(rt - 0.2)/0.2 * 100.0, 0.0, 100.0))
    
    dt = validated_features.get("decay_time_sec", 0.5)
    norm_dt = float(np.clip(100.0 - abs(dt - 0.5)/0.5 * 100.0, 0.0, 100.0))
    
    morphology_radar = {
        "Heart Rate Centroid": norm_hr,
        "Systolic Pulse Width": norm_pw,
        "Anacrotic Rise Phase": norm_rt,
        "Dicrotic Decay Phase": norm_dt
    }
    
    # 5. Artifact Heatmap Matrix
    # Divide signal into 10 temporal chunks and evaluate noise variables in each
    chunks_count = 10
    chunk_size = n // chunks_count
    heatmap_matrix = []
    
    for i in range(chunks_count):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        if end - start < 10:
            continue
            
        chunk_raw = raw_signal[start:end]
        chunk_filt = preprocessed_signal[start:end]
        
        # Calculate local noise metrics
        c_diff = np.diff(chunk_filt)
        c_diff_std = np.std(c_diff)
        c_motion = float(np.sum(np.abs(c_diff) > 4.0 * c_diff_std) / len(c_diff) * 100.0) if c_diff_std > 1e-9 else 0.0
        
        c_flat = float(np.sum(np.abs(c_diff) < 1e-7) / len(c_diff) * 100.0)
        
        # Baseline drift ratio locally
        c_std = np.std(chunk_filt)
        # Detrended standard deviation vs raw standard deviation
        c_detrend = scipy.signal.detrend(chunk_filt)
        c_drift = float((c_std - np.std(c_detrend)) / (c_std + 1e-9) * 100.0)
        c_drift = max(0.0, c_drift)
        
        # Row layout: [Motion Outliers %, Baseline Wander %, Flatline %]
        heatmap_matrix.append([c_motion, c_drift, c_flat])
        
    return {
        "signal_data": {
            "time": time_array.tolist(),
            "raw": raw_signal.tolist(),
            "preprocessed": preprocessed_signal.tolist()
        },
        "peaks": peaks.tolist(),
        "derivatives": {
            "vpg": vpg.tolist(),
            "apg": apg.tolist()
        },
        "trends": {
            "rr_times_sec": rr_times.tolist(),
            "rr_intervals_sec": rr_intervals.tolist(),
            "hr_trend_bpm": hr_trend.tolist()
        },
        "radar": {
            "quality": quality_radar,
            "morphology": morphology_radar
        },
        "heatmap": {
            "matrix": heatmap_matrix,
            "y_labels": [f"Chunk {i}" for i in range(len(heatmap_matrix))],
            "x_labels": ["Motion Artifacts", "Baseline Wander", "Flat Connection"]
        }
    }
