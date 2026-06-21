import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple
from biopulse_backend.core.filter_playground import run_filter, butterworth_filter, chebyshev_filter, savitzky_golay_filter, gaussian_filter, median_filter

def extract_morphology_properties(sig: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extracts key physiological morphology features from the signal.
    """
    sig_detrend = scipy.signal.detrend(sig)
    std_val = np.std(sig_detrend)
    dist = max(5, int(fs * 0.4))
    
    peaks, props = scipy.signal.find_peaks(sig_detrend, distance=dist, prominence=0.05 * std_val)
    troughs, _ = scipy.signal.find_peaks(-sig_detrend, distance=dist, prominence=0.05 * std_val)
    
    if len(peaks) < 2:
        return {
            "peak_count": float(len(peaks)),
            "hr": 72.0,
            "rr": 0.833,
            "pulse_width": 0.3,
            "rise_time": 0.25,
            "decay_time": 0.58,
            "amplitude": float(std_val * 2.0) if std_val > 0 else 1.0,
            "area": float(std_val) if std_val > 0 else 0.5,
            "prominence": float(std_val) if std_val > 0 else 0.5
        }
        
    # RR intervals (seconds)
    rr_intervals = np.diff(peaks) / fs
    rr_mean = float(np.mean(rr_intervals))
    hr = 60.0 / rr_mean
    
    # Prominence
    prominences = props.get("prominences", np.array([std_val]))
    mean_prom = float(np.mean(prominences))
    
    # Pulse widths at half-height (seconds)
    widths, _, _, _ = scipy.signal.peak_widths(sig_detrend, peaks, rel_height=0.5)
    mean_width = float(np.mean(widths) / fs)
    
    # Rise and decay times
    rise_times = []
    decay_times = []
    for p in peaks:
        t_before = troughs[troughs < p]
        if len(t_before) > 0:
            rise_times.append((p - t_before[-1]) / fs)
        t_after = troughs[troughs > p]
        if len(t_after) > 0:
            decay_times.append((t_after[0] - p) / fs)
            
    mean_rise = float(np.mean(rise_times)) if rise_times else rr_mean * 0.3
    mean_decay = float(np.mean(decay_times)) if decay_times else rr_mean * 0.7
    
    # Amplitude: peak to trough
    mean_peak_val = np.mean(sig[peaks]) if len(peaks) > 0 else np.mean(sig)
    mean_trough_val = np.mean(sig[troughs]) if len(troughs) > 0 else np.mean(sig)
    amplitude = float(mean_peak_val - mean_trough_val)
    if amplitude <= 0:
        amplitude = float(std_val * 2.0)
        
    # Area
    area = float(np.sum(np.abs(sig_detrend)) / fs / len(peaks))
    
    return {
        "peak_count": float(len(peaks)),
        "hr": hr,
        "rr": rr_mean,
        "pulse_width": mean_width,
        "rise_time": mean_rise,
        "decay_time": mean_decay,
        "amplitude": amplitude,
        "area": area,
        "prominence": mean_prom
    }

def run_filter_at_strength(filter_name: str, signal: np.ndarray, fs: float, strength_level: int) -> np.ndarray:
    """
    Applies the named filter with strength adjusted by strength_level (0 = standard, 1 = weaker, 2 = very weak, 3 = bypass).
    """
    if strength_level >= 3:
        return signal.copy()
        
    name = filter_name.lower()
    
    if name == "none":
        return signal.copy()
        
    elif name == "butterworth":
        # Adjust cutoffs and order
        if strength_level == 0:
            return butterworth_filter(signal, fs, order=3, lowcut=0.5, highcut=5.0)
        elif strength_level == 1:
            return butterworth_filter(signal, fs, order=2, lowcut=0.3, highcut=8.0)
        else:
            return butterworth_filter(signal, fs, order=1, lowcut=0.1, highcut=12.0)
            
    elif name == "chebyshev":
        if strength_level == 0:
            return chebyshev_filter(signal, fs, order=3, rs=20.0, lowcut=0.5, highcut=5.0)
        elif strength_level == 1:
            return chebyshev_filter(signal, fs, order=2, rs=30.0, lowcut=0.3, highcut=8.0)
        else:
            return chebyshev_filter(signal, fs, order=1, rs=40.0, lowcut=0.1, highcut=12.0)
            
    elif name == "savitzky-golay":
        # SG: decrease window size to reduce smoothing strength
        if strength_level == 0:
            return savitzky_golay_filter(signal, fs, polyorder=2, window_size_sec=0.4)
        elif strength_level == 1:
            return savitzky_golay_filter(signal, fs, polyorder=2, window_size_sec=0.2)
        else:
            return savitzky_golay_filter(signal, fs, polyorder=2, window_size_sec=0.1)
            
    elif name == "gaussian":
        # Gaussian: decrease sigma to reduce smoothing strength
        if strength_level == 0:
            return gaussian_filter(signal, fs, sigma_sec=0.05)
        elif strength_level == 1:
            return gaussian_filter(signal, fs, sigma_sec=0.025)
        else:
            return gaussian_filter(signal, fs, sigma_sec=0.01)
            
    elif name == "median":
        if strength_level == 0:
            return median_filter(signal, fs, kernel_size_sec=0.1)
        elif strength_level == 1:
            return median_filter(signal, fs, kernel_size_sec=0.05)
        else:
            return median_filter(signal, fs, kernel_size_sec=0.03)
            
    elif name == "wavelet":
        # We can stub wavelet adjustment by scaling the threshold down
        return run_filter("wavelet", signal, fs)
        
    return signal.copy()

def guard_filter_morphology(
    signal: np.ndarray,
    fs: float,
    filter_name: str
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Applies the filter and verifies that deviations do not exceed thresholds.
    If thresholds are exceeded, it retry up to 3 times with reduced strength.
    Returns:
        filtered_signal: output signal
        degraded: True if we had to reduce strength or fallback to raw
        details: dictionary with comparison scores per attempt
    """
    raw_props = extract_morphology_properties(signal, fs)
    
    degraded = False
    details = {}
    
    # Retry loop
    for attempt in range(4):
        filtered = run_filter_at_strength(filter_name, signal, fs, attempt)
        filt_props = extract_morphology_properties(filtered, fs)
        
        # Calculate deviations
        # Avoid division by zero
        hr_raw = max(1.0, raw_props["hr"])
        pk_raw = max(1.0, raw_props["peak_count"])
        rr_raw = max(0.01, raw_props["rr"])
        pw_raw = max(0.01, raw_props["pulse_width"])
        amp_raw = max(1e-9, raw_props["amplitude"])
        
        hr_dev = abs(raw_props["hr"] - filt_props["hr"]) / hr_raw
        pk_dev = abs(raw_props["peak_count"] - filt_props["peak_count"]) / pk_raw
        rr_dev = abs(raw_props["rr"] - filt_props["rr"]) / rr_raw
        pw_dev = abs(raw_props["pulse_width"] - filt_props["pulse_width"]) / pw_raw
        amp_dev = abs(raw_props["amplitude"] - filt_props["amplitude"]) / amp_raw
        
        # Thresholds: HR < 3%, Peak count < 5%, RR < 5%, pulse width < 5%, amplitude < 10%
        hr_pass = hr_dev < 0.03
        pk_pass = pk_dev < 0.05
        rr_pass = rr_dev < 0.05
        pw_pass = pw_dev < 0.05
        amp_pass = amp_dev < 0.10
        
        passed = hr_pass and pk_pass and rr_pass and pw_pass and amp_pass
        
        attempt_details = {
            "hr_deviation": hr_dev,
            "peak_count_deviation": pk_dev,
            "rr_deviation": rr_dev,
            "pulse_width_deviation": pw_dev,
            "amplitude_deviation": amp_dev,
            "passed": passed
        }
        details[f"attempt_{attempt}"] = attempt_details
        
        if passed:
            degraded = (attempt > 0)
            details["final_strength_attempt"] = attempt
            details["status"] = "PASSED"
            return filtered, degraded, details
            
    # If all attempts fail, we fallback to the raw signal (attempt 3) and mark as degraded
    details["final_strength_attempt"] = 3
    details["status"] = "DEGRADED_FALLBACK"
    return signal.copy(), True, details
