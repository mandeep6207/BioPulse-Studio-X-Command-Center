import numpy as np
import scipy.signal
from scipy.stats import skew
from typing import Tuple

def detect_orientation(signal: np.ndarray, fs: float) -> Tuple[str, bool, float]:
    """
    Detects if the PPG signal is inverted.
    Returns:
        orientation: 'normal', 'inverted', or 'uncertain'
        recommend_reverse: True if we recommend inverting, False otherwise
        confidence: float representing detection confidence (0.0 to 1.0)
    """
    # 1. Detrend signal first to focus on AC component
    sig_detrend = scipy.signal.detrend(signal)
    
    # 2. Skewness check
    sig_skew = float(skew(sig_detrend))
    
    # 3. Peak prominence ratio check
    # Find peaks on original signal
    std_val = np.std(sig_detrend)
    if std_val < 1e-9:
        return "uncertain", False, 0.0
        
    dist = max(5, int(fs * 0.35)) # typical min beat distance (approx 170 bpm)
    
    peaks_pos, props_pos = scipy.signal.find_peaks(sig_detrend, distance=dist, prominence=0.1 * std_val)
    peaks_neg, props_neg = scipy.signal.find_peaks(-sig_detrend, distance=dist, prominence=0.1 * std_val)
    
    prom_pos = np.mean(props_pos["prominences"]) if len(peaks_pos) > 0 else 0.0
    prom_neg = np.mean(props_neg["prominences"]) if len(peaks_neg) > 0 else 0.0
    
    # In a standard oriented PPG (blood absorption peaks pointing upwards or standard optical signal),
    # the systolic peaks are sharper and more prominent.
    # We compare the ratio of positive peak prominence to negative peak prominence.
    prom_ratio = 1.0
    if prom_pos > 0 and prom_neg > 0:
        prom_ratio = prom_pos / prom_neg
    elif prom_pos > 0:
        prom_ratio = 5.0
    elif prom_neg > 0:
        prom_ratio = 0.2
        
    # Decision logic
    # Positive skewness and higher positive peak prominence indicate normal orientation
    # Negative skewness and higher negative peak prominence indicate inverted orientation
    
    # Let's map these indicators to a confidence score
    normal_score = 0.0
    inverted_score = 0.0
    
    # Skewness contributions
    if sig_skew > 0.15:
        normal_score += min(1.0, sig_skew * 2.0)
    elif sig_skew < -0.15:
        inverted_score += min(1.0, -sig_skew * 2.0)
        
    # Prominence ratio contributions
    if prom_ratio > 1.2:
        normal_score += min(1.0, (prom_ratio - 1.0) * 1.5)
    elif prom_ratio < 0.8:
        inverted_score += min(1.0, (1.0 / (prom_ratio + 1e-5) - 1.0) * 1.5)
        
    # Combine and normalize
    total = normal_score + inverted_score
    if total == 0:
        orientation = "uncertain"
        recommend_reverse = False
        confidence = 0.0
    else:
        diff = abs(normal_score - inverted_score)
        confidence = float(min(1.0, diff / 2.0))
        
        # If confidence is below 0.35, we classify as uncertain
        if confidence < 0.35:
            orientation = "uncertain"
            recommend_reverse = (sig_skew < 0) # recommend reversing if skew is negative
        else:
            if normal_score > inverted_score:
                orientation = "normal"
                recommend_reverse = False
            else:
                orientation = "inverted"
                recommend_reverse = True
                
    return orientation, recommend_reverse, confidence
