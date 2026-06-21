import os
import numpy as np
from typing import Dict, Any, Tuple

def classify_signal(signal: np.ndarray, fs: float, file_info: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[str, str, float]:
    """
    Classifies a PPG signal by duration and guesses its source.
    Returns:
        duration_bucket: '<20s Short', '20-120s Standard', '120-300s Long', '>300s Extended'
        source_guess: 'Mobile', 'Clinical', 'Research', 'Unknown'
        duration_seconds: calculated duration in seconds
    """
    n = len(signal)
    duration_seconds = n / fs
    
    # 1. Determine duration bucket
    if duration_seconds < 20.0:
        duration_bucket = "<20s Short"
    elif duration_seconds <= 120.0:
        duration_bucket = "20-120s Standard"
    elif duration_seconds <= 300.0:
        duration_bucket = "120-300s Long"
    else:
        duration_bucket = ">300s Extended"
        
    # 2. Source guessing heuristics
    filename = file_info.get("file_name", "").lower()
    
    # Check amplitude range
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    amp_range = signal_max - signal_min
    
    source_guess = "Unknown"
    
    # Check filename keywords
    if any(kw in filename for kw in ["mobile", "phone", "camera", "app", "sensor", "smartwatch"]):
        source_guess = "Mobile"
    elif any(kw in filename for kw in ["clg", "aiims", "clinical", "patient", "hosp", "subject"]):
        if amp_range > 10.0:
            # High values with Clg could be Mobile-captured PPG text files
            source_guess = "Mobile"
        else:
            source_guess = "Clinical"
    elif any(kw in filename for kw in ["sub", "research", "lab", "trial"]):
        source_guess = "Research"
    # Check metadata markers
    elif any("acq" in str(k).lower() or "acq" in str(v).lower() for k, v in metadata.items()):
        source_guess = "Research"
    elif any("channels" in str(k).lower() or "channels" in str(v).lower() for k, v in metadata.items()):
        source_guess = "Research"
    # Check typical amplitude / FS traits
    else:
        if fs <= 60.0:
            source_guess = "Mobile"
        elif fs >= 200.0 and amp_range < 5.0:
            # Low voltage, high sampling rate (e.g. mV biopac or clinical ADC)
            source_guess = "Clinical"
        elif fs >= 100.0 and amp_range > 10.0:
            source_guess = "Research"
            
    return duration_bucket, source_guess, float(duration_seconds)
