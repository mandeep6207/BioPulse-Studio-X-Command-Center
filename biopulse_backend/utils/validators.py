import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

def validate_raw_signal(signal: Any) -> np.ndarray:
    """
    Validates a raw input signal, converting it to a 1D float64 numpy array.
    Raises ValueError if the signal is unrecoverable (empty, contains only NaNs, or not 1D).
    """
    if signal is None:
        raise ValueError("Signal cannot be None")
        
    if isinstance(signal, (list, tuple)):
        signal = np.array(signal, dtype=np.float64)
    elif isinstance(signal, pd.Series):
        signal = signal.to_numpy(dtype=np.float64)
    elif isinstance(signal, pd.DataFrame):
        if signal.shape[1] == 1:
            signal = signal.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"DataFrame input must have exactly 1 column, got {signal.shape[1]}")
    elif not isinstance(signal, np.ndarray):
        try:
            signal = np.array(signal, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Could not convert input to numpy array: {str(e)}")
            
    # Flatten if needed
    signal = signal.squeeze()
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {signal.shape}")
        
    if len(signal) == 0:
        raise ValueError("Signal is empty")
        
    # Check if all values are NaN or Inf
    if np.all(np.isnan(signal)) or np.all(np.isinf(signal)):
        raise ValueError("Signal contains only NaNs or Infs and is unrecoverable")
        
    return signal.astype(np.float64)

def check_parameters(fs: float, limits: Dict[str, Any]) -> None:
    """
    Validates that sampling rate and other controls are within physiological/logical limits.
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if fs > 10000:
        raise ValueError(f"Sampling frequency {fs} is too high (max 10000 Hz)")
