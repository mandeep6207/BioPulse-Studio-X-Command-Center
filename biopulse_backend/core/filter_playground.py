import numpy as np
import scipy.signal
import scipy.ndimage
import pywt
from typing import Dict, Any, Tuple, List
from biopulse_backend.utils.math_utils import calculate_snr_and_drift
from biopulse_backend.core.quality_engine import calculate_beat_consistency

# 1. Butterworth filter
def butterworth_filter(signal: np.ndarray, fs: float, order: int = 3, lowcut: float = 0.5, highcut: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp cutoff frequencies to avoid nyquist errors
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, 0.002, 0.999)
    if low >= high:
        low = high / 2.0
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b, a, signal)

# 2. Chebyshev Type II filter
def chebyshev_filter(signal: np.ndarray, fs: float, order: int = 3, rs: float = 20.0, lowcut: float = 0.5, highcut: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, 0.002, 0.999)
    if low >= high:
        low = high / 2.0
    b, a = scipy.signal.cheby2(order, rs, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b, a, signal)

# 3. Savitzky-Golay filter
def savitzky_golay_filter(signal: np.ndarray, fs: float, polyorder: int = 2, window_size_sec: float = 0.4) -> np.ndarray:
    window_len = int(window_size_sec * fs)
    if window_len % 2 == 0:
        window_len += 1
    # Adjust if signal is too short or window is too small
    window_len = max(5, min(window_len, len(signal) - 1))
    # Make sure window_len is odd and greater than polyorder
    if window_len <= polyorder:
        window_len = polyorder + 1
        if window_len % 2 == 0:
            window_len += 1
    return scipy.signal.savgol_filter(signal, window_len, polyorder)

# 4. Gaussian filter
def gaussian_filter(signal: np.ndarray, fs: float, sigma_sec: float = 0.05) -> np.ndarray:
    sigma = sigma_sec * fs
    sigma = max(0.5, sigma)
    return scipy.ndimage.gaussian_filter1d(signal, sigma)

# 5. Median filter
def median_filter(signal: np.ndarray, fs: float, kernel_size_sec: float = 0.1) -> np.ndarray:
    kernel_size = int(kernel_size_sec * fs)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, len(signal) - 1))
    return scipy.signal.medfilt(signal, kernel_size)

# 6. Wavelet filter
def wavelet_filter(signal: np.ndarray) -> np.ndarray:
    wavelet = 'db4'
    level = min(4, pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len))
    if level == 0:
        return signal.copy()
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Universal threshold: sigma * sqrt(2 * log(N))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2.0 * np.log(len(signal)))
        new_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        filtered = pywt.waverec(new_coeffs, wavelet)
        return filtered[:len(signal)]
    except Exception:
        return signal.copy()

def run_filter(filter_name: str, signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Applies the named filter to the signal.
    """
    name = filter_name.lower()
    if name == "none":
        return signal.copy()
    elif name == "butterworth":
        return butterworth_filter(signal, fs)
    elif name == "chebyshev":
        return chebyshev_filter(signal, fs)
    elif name == "savitzky-golay":
        return savitzky_golay_filter(signal, fs)
    elif name == "gaussian":
        return gaussian_filter(signal, fs)
    elif name == "median":
        return median_filter(signal, fs)
    elif name == "wavelet":
        return wavelet_filter(signal)
    else:
        return signal.copy()

def evaluate_filters(signal: np.ndarray, fs: float) -> Tuple[str, bool, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Applies all 7 filters and scores them on Morphology (40%), Peak Retention (30%),
    Beat Similarity (20%), and SNR (10%).
    Returns:
        best_filter: name of the top-performing filter
        needs_comparison: True if the top filters are within 3% of each other
        scoreboard: detailed scores per filter
        filtered_signals: dictionary of filtered signal arrays
    """
    filters = ["None", "Butterworth", "Chebyshev", "Savitzky-Golay", "Gaussian", "Median", "Wavelet"]
    scoreboard = {}
    filtered_signals = {}
    
    # Detrend raw for peak finding
    raw_detrend = scipy.signal.detrend(signal)
    dist = max(5, int(fs * 0.4))
    raw_peaks, _ = scipy.signal.find_peaks(raw_detrend, distance=dist, prominence=0.1 * np.std(raw_detrend))
    raw_snr, _ = calculate_snr_and_drift(signal, fs)
    
    for f_name in filters:
        filtered = run_filter(f_name, signal, fs)
        filtered_signals[f_name] = filtered
        
        # 1. Morphology Score (Pearson correlation)
        # Handle constant signal
        if np.std(signal) < 1e-9 or np.std(filtered) < 1e-9:
            morphology = 0.0
        else:
            r = np.corrcoef(signal, filtered)[0, 1]
            morphology = max(0.0, float(r)) * 100.0
            
        # 2. Peak Retention Score
        filt_detrend = scipy.signal.detrend(filtered)
        filt_peaks, _ = scipy.signal.find_peaks(filt_detrend, distance=dist, prominence=0.1 * np.std(filt_detrend))
        
        peak_diff = abs(len(raw_peaks) - len(filt_peaks))
        raw_peak_count = max(1, len(raw_peaks))
        peak_retention = max(0.0, 100.0 - (peak_diff / raw_peak_count * 100.0))
        
        # 3. Beat Similarity (Consistency)
        consistency = calculate_beat_consistency(filt_detrend, fs, filt_peaks)
        beat_similarity = consistency * 100.0
        
        # 4. SNR Improvement
        filt_snr, _ = calculate_snr_and_drift(filtered, fs)
        if filt_snr > raw_snr:
            snr_score = min(100.0, 50.0 + (filt_snr - raw_snr) * 10.0)
        else:
            snr_score = max(0.0, 50.0 - (raw_snr - filt_snr) * 10.0)
            
        # Unified score
        total_score = 0.40 * morphology + 0.30 * peak_retention + 0.20 * beat_similarity + 0.10 * snr_score
        
        scoreboard[f_name] = {
            "morphology": morphology,
            "peak_retention": peak_retention,
            "beat_similarity": beat_similarity,
            "snr_score": snr_score,
            "total_score": total_score,
            "snr_db": filt_snr
        }
        
    # Find the best filter
    best_filter = "None"
    best_score = -1.0
    for f_name, metrics in scoreboard.items():
        if metrics["total_score"] > best_score:
            best_score = metrics["total_score"]
            best_filter = f_name
            
    # Check if top filters are within 3% of each other (needs comparison flag)
    top_scores = sorted([m["total_score"] for m in scoreboard.values()], reverse=True)
    needs_comparison = False
    if len(top_scores) > 1:
        if (top_scores[0] - top_scores[1]) <= 3.0:
            needs_comparison = True
            
    return best_filter, needs_comparison, scoreboard, filtered_signals
