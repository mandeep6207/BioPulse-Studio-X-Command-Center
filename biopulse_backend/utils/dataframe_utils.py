import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

def detect_encoding(file_path: str) -> str:
    """
    Tries to decode the file using a chain of encodings: utf-8, utf-16, latin-1.
    Returns the first encoding that succeeds without UnicodeDecodeError.
    """
    encodings = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read(1024)
            return enc
        except Exception:
            continue
    return 'utf-8' # Default fallback

def detect_delimiter(file_path: str, encoding: str) -> str:
    """
    Analyzes the first 20 lines to count occurrences of different delimiters: , ; tab whitespace.
    Returns the most likely delimiter.
    """
    delimiters = [',', ';', '\t', ' ']
    counts = {d: 0 for d in delimiters}
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            lines = [f.readline() for _ in range(20)]
            lines = [l for l in lines if l.strip()]
            
        if not lines:
            return ','
            
        for line in lines:
            for d in delimiters:
                counts[d] += line.count(d)
                
        # Handle whitespace vs single space
        # If tabs are found, prioritize tab
        if counts['\t'] > 0:
            return '\t'
            
        # Select delimiter with maximum count
        best_delim = max(counts, key=counts.get)
        if counts[best_delim] == 0:
            # If no delimiter found, might be single column whitespace or newlines
            return ','
        return best_delim
    except Exception:
        return ','

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs missing value handling: linear interpolation for internal NaNs,
    and forward/backward fill for boundary NaNs.
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Perform linear interpolation column-wise
    df = df.interpolate(method='linear', limit_direction='both')
    # Final fallback for empty/all NaN columns
    df = df.fillna(0.0)
    return df
