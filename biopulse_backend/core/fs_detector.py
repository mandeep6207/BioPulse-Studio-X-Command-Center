import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from biopulse_backend.utils.math_utils import estimate_fs_from_data

def detect_sampling_frequency(
    signal: np.ndarray,
    file_info: Dict[str, Any],
    metadata: Dict[str, Any],
    df: pd.DataFrame,
    user_fs: Optional[float] = None
) -> Tuple[float, float, str, Dict[str, Any]]:
    """
    Detects the sampling rate using the priority chain:
    User Override -> Metadata -> Timestamp-derived -> Filename Hint -> Estimation.
    Computes agreement score using DSP voting and returns status (PASS/WARNING/FS_UNRESOLVED).
    """
    selected_fs = None
    source = "Estimation"
    
    # 1. User override
    if user_fs is not None and user_fs > 0:
        selected_fs = float(user_fs)
        source = "User Override"
        
    # 2. Metadata parsing
    if selected_fs is None:
        # Check metadata dictionary for sample rate keywords
        for k, v in metadata.items():
            k_str, v_str = str(k).lower(), str(v).lower()
            
            # Match Biopac style: "X msec/sample" or "X ms/sample"
            msec_match = re.search(r"(\d+(?:\.\d+)?)\s*m?sec/sample", k_str + " " + v_str)
            if msec_match:
                ms_val = float(msec_match.group(1))
                if ms_val > 0:
                    selected_fs = 1000.0 / ms_val
                    source = "Metadata (msec/sample)"
                    break
                    
            # Match standard Hz: "sample rate: 250", "fs: 100 Hz"
            hz_match = re.search(r"(?:sample\s*rate|sampling\s*rate|sampling\s*freq|fs|frequency)\b.*?(\d+(?:\.\d+)?)\s*(?:hz|samples/sec)?", k_str + " " + v_str)
            if hz_match:
                hz_val = float(hz_match.group(1))
                if hz_val > 0:
                    selected_fs = hz_val
                    source = "Metadata (Keywords)"
                    break
                    
    # 3. Timestamp-derived
    if selected_fs is None and file_info.get("timestamp_col") is not None:
        t_col = file_info["timestamp_col"]
        time_vals = df[t_col].to_numpy()
        if len(time_vals) > 1:
            diffs = np.diff(time_vals)
            dt = float(np.mean(diffs))
            if dt > 0:
                t_col_lower = str(t_col).lower()
                # Check unit
                if "min" in t_col_lower:
                    dt_sec = dt * 60.0
                elif "ms" in t_col_lower or "msec" in t_col_lower:
                    dt_sec = dt / 1000.0
                else:
                    # Guess unit based on range: if max time is very small, it might be minutes
                    # E.g. Sub2 has max time of 5.0 (for 300000 samples at 1000 Hz, 300 seconds = 5 minutes).
                    # If max_time * 60 is close to length / 100 or length / 1000, we convert.
                    max_t = time_vals[-1]
                    if max_t < len(time_vals) / 10.0 and max_t > 0:
                        # Likely minutes or hours
                        dt_sec = dt * 60.0
                    else:
                        dt_sec = dt
                        
                selected_fs = 1.0 / dt_sec
                source = "Timestamp-derived"
                
    # 4. Filename hints
    if selected_fs is None:
        filename = file_info.get("file_name", "").lower()
        # Find e.g. "100hz", "250hz", "500hz", "30hz"
        fn_match = re.search(r"(\d+)\s*hz", filename)
        if fn_match:
            selected_fs = float(fn_match.group(1))
            source = "Filename Hint"
            
    # Run data-driven estimation for verification / voting
    est_results = estimate_fs_from_data(signal)
    estimated_fs = est_results["estimated_fs"]
    agreement_score = est_results["agreement_score"]
    votes = est_results["votes"]
    
    # 5. Estimation fallback
    if selected_fs is None:
        selected_fs = estimated_fs
        source = "Estimation"
        confidence = agreement_score
    else:
        # If we had a direct source, evaluate its agreement with the estimated Fs
        # If they are within 10% of each other, confidence is high, else warning.
        pct_diff = abs(selected_fs - estimated_fs) / selected_fs
        if pct_diff < 0.10:
            confidence = 1.0
        else:
            confidence = max(0.5, 1.0 - pct_diff)
            
    # Categorize status based on confidence
    if confidence > 0.90:
        status = "PASS"
    elif confidence >= 0.70:
        status = "WARNING"
    else:
        status = "FS_UNRESOLVED"
        
    details = {
        "source": source,
        "estimated_fs": estimated_fs,
        "agreement_score": agreement_score,
        "votes": votes,
        "confidence": confidence,
        "estimation_details": est_results
    }
    
    return float(selected_fs), float(confidence), status, details
