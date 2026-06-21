import numpy as np
import scipy.signal
from typing import Tuple, Dict, Any
from biopulse_backend.core.filter_playground import run_filter
from biopulse_backend.utils.math_utils import calculate_derivatives

def count_inflection_points(signal: np.ndarray, fs: float, peaks: np.ndarray) -> float:
    """
    Counts the average number of local peaks in the 2nd derivative (APG) per heartbeat cycle.
    PPG with dicrotic notch exhibits multiple waves (a, b, c, d, e), hence >= 2 APG peaks per beat.
    """
    if len(peaks) < 3:
        return 1.0
        
    _, apg = calculate_derivatives(signal, fs)
    apg_peaks_count = []
    
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i+1]
        beat_apg = apg[start:end]
        if len(beat_apg) > 5:
            # Find local peaks in the APG wave for this beat
            # Standard threshold: small prominence to catch notches
            apg_peaks, _ = scipy.signal.find_peaks(beat_apg, prominence=0.01 * np.std(beat_apg))
            apg_peaks_count.append(len(apg_peaks))
            
    if not apg_peaks_count:
        return 1.0
    return float(np.mean(apg_peaks_count))

def preprocess_signal(
    signal: np.ndarray, 
    fs: float, 
    filter_name: str
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Applies exactly ONE primary filter to the signal and evaluates whether the dicrotic
    notch and high-frequency morphology are preserved.
    """
    # 1. Apply single filter
    filtered = run_filter(filter_name, signal, fs)
    
    # 2. Check notch preservation
    # Detrend for peak finding
    raw_detrend = scipy.signal.detrend(signal)
    filt_detrend = scipy.signal.detrend(filtered)
    
    dist = max(5, int(fs * 0.4))
    raw_peaks, _ = scipy.signal.find_peaks(raw_detrend, distance=dist, prominence=0.1 * np.std(raw_detrend))
    filt_peaks, _ = scipy.signal.find_peaks(filt_detrend, distance=dist, prominence=0.1 * np.std(filt_detrend))
    
    raw_apg_peaks = count_inflection_points(signal, fs, raw_peaks)
    filt_apg_peaks = count_inflection_points(filtered, fs, filt_peaks)
    
    # If the raw signal has multi-wave APG properties (mean peaks >= 1.5)
    # and the filtered signal maintains at least 80% of these inflection points,
    # or if both have similar numbers, we classify it as notch preserved.
    notch_preserved = True
    if raw_apg_peaks >= 1.5:
        # If it was complex, but became too simple (under 1.2 peaks on average)
        if filt_apg_peaks < 1.2:
            notch_preserved = False
            
    details = {
        "raw_apg_peaks_per_beat": raw_apg_peaks,
        "filtered_apg_peaks_per_beat": filt_apg_peaks,
        "applied_filter": filter_name,
        "notch_preserved": notch_preserved
    }
    
    return filtered, notch_preserved, details
