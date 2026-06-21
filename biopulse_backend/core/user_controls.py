import numpy as np
import scipy.signal
from typing import Tuple, List, Optional

def apply_fs_override(fs: float, fs_override: Optional[float] = None) -> float:
    if fs_override is not None and fs_override > 0:
        return float(fs_override)
    return fs

def apply_reverse(signal: np.ndarray, reverse: bool = False) -> np.ndarray:
    if reverse:
        return -signal
    return signal

def apply_trim(
    signal: np.ndarray, 
    fs: float, 
    trim_start_sec: float = 0.0, 
    trim_end_sec: float = 0.0, 
    time_series: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = len(signal)
    start_idx = int(trim_start_sec * fs)
    end_idx = n - int(trim_end_sec * fs)
    
    # Clamp bounds
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx + 1, min(end_idx, n))
    
    trimmed_signal = signal[start_idx:end_idx]
    trimmed_time = time_series[start_idx:end_idx] if time_series is not None else None
    
    return trimmed_signal, trimmed_time

def apply_centering(signal: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """
    Centers or detrends the signal.
    Methods: 'mean', 'median', 'detrend', None.
    """
    if method == "mean":
        return signal - np.mean(signal)
    elif method == "median":
        return signal - np.median(signal)
    elif method == "detrend":
        return scipy.signal.detrend(signal)
    return signal

def apply_chunking(
    signal: np.ndarray, 
    fs: float, 
    chunk_size_sec: float = 30.0, 
    time_series: Optional[np.ndarray] = None
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Splits the signal (and optionally time_series) into non-overlapping windows of chunk_size_sec.
    """
    chunk_samples = int(chunk_size_sec * fs)
    if chunk_samples <= 0 or chunk_samples > len(signal):
        return [(signal, time_series)]
        
    chunks = []
    n = len(signal)
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        # Avoid very short tail chunks (< 1 second)
        if (end - start) < int(fs):
            if chunks: # Append tail to the last chunk
                last_sig, last_t = chunks[-1]
                updated_sig = np.concatenate([last_sig, signal[start:end]])
                updated_t = np.concatenate([last_t, time_series[start:end]]) if time_series is not None else None
                chunks[-1] = (updated_sig, updated_t)
            else:
                chunks.append((signal[start:end], time_series[start:end] if time_series is not None else None))
            break
            
        chunk_sig = signal[start:end]
        chunk_t = time_series[start:end] if time_series is not None else None
        chunks.append((chunk_sig, chunk_t))
        
    return chunks
