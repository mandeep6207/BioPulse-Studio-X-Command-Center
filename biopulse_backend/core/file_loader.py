import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from biopulse_backend.utils.dataframe_utils import detect_encoding, detect_delimiter, sanitize_dataframe

def is_float(val: str) -> bool:
    try:
        float(val.strip())
        return True
    except ValueError:
        return False

def load_ppg_file(file_path: str, signal_col_override: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Loads a PPG file, auto-detecting delimiters, headers, metadata, and key columns.
    Returns:
        df: sanitized pandas DataFrame of numeric data
        metadata: dictionary of extracted metadata lines
        file_info: dictionary with file_path, detected_encoding, delimiter, signal_col, etc.
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    
    # Read all lines to identify metadata, header, and data start
    with open(file_path, 'r', encoding=encoding) as f:
        raw_lines = f.readlines()
        
    # Find mode of column count in the file
    col_counts = []
    for line in raw_lines:
        line_str = line.strip()
        if not line_str:
            continue
        tokens = line_str.split(delimiter) if delimiter != ' ' else line_str.split()
        # Remove trailing empty token if delimiter is trailing
        if tokens and tokens[-1] == '':
            tokens = tokens[:-1]
        col_counts.append(len(tokens))
        
    if not col_counts:
        raise ValueError(f"No delimited data found in file {file_path}")
        
    from collections import Counter
    expected_cols = Counter(col_counts).most_common(1)[0][0]
    
    # Find first row index where tabular numeric data begins
    data_start_idx = -1
    
    for i, line in enumerate(raw_lines):
        line_str = line.strip()
        if not line_str:
            continue
        tokens = line_str.split(delimiter) if delimiter != ' ' else line_str.split()
        if tokens and tokens[-1] == '':
            tokens = tokens[:-1]
            
        if len(tokens) == expected_cols:
            # Check if this and next 5 lines are mostly numeric
            numeric_check = True
            check_range = min(len(raw_lines), i + 5)
            for j in range(i, check_range):
                j_line = raw_lines[j].strip()
                if not j_line:
                    continue
                j_tokens = j_line.split(delimiter) if delimiter != ' ' else j_line.split()
                if j_tokens and j_tokens[-1] == '':
                    j_tokens = j_tokens[:-1]
                if len(j_tokens) != expected_cols:
                    numeric_check = False
                    break
                # Try to parse at least one token as float
                parsed_count = 0
                for tok in j_tokens:
                    if is_float(tok):
                        parsed_count += 1
                if parsed_count < expected_cols - 1: # allow header or ID column
                    numeric_check = False
                    break
            
            if numeric_check:
                data_start_idx = i
                break
                
    if data_start_idx == -1:
        data_start_idx = 0
        
    # Search upwards from data_start_idx - 1 to find the header row
    header_idx = -1
    suspicious_indices = []
    
    if data_start_idx > 0:
        for check_idx in range(data_start_idx - 1, max(-1, data_start_idx - 4), -1):
            check_line = raw_lines[check_idx].strip()
            if not check_line:
                continue
            tokens = check_line.split(delimiter) if delimiter != ' ' else check_line.split()
            if tokens and tokens[-1] == '':
                tokens = tokens[:-1]
                
            if len(tokens) == expected_cols:
                # Check if it contains mostly non-numeric tokens
                non_numeric = sum(1 for tok in tokens if not is_float(tok))
                is_suspicious = (len(tokens) > 0 and tokens[0] == '') or any(
                    'sample' in str(tok).lower() or 'count' in str(tok).lower() for tok in tokens
                )
                
                if non_numeric >= expected_cols // 2 or expected_cols == 1:
                    if not is_suspicious:
                        header_idx = check_idx
                        break
                    else:
                        suspicious_indices.append(check_idx)
                        
    # Parse metadata rows
    metadata = {}
    metadata_end_idx = header_idx if header_idx != -1 else (min(suspicious_indices) if suspicious_indices else data_start_idx)
    for idx in range(metadata_end_idx):
        line = raw_lines[idx].strip()
        if not line:
            continue
        if ":" in line:
            parts = line.split(":", 1)
            metadata[parts[0].strip()] = parts[1].strip()
        else:
            metadata[f"meta_line_{idx}"] = line
            
    # Setup row skipping list
    skip_rows = list(range(metadata_end_idx)) + suspicious_indices
    
    try:
        if header_idx != -1:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter if delimiter != ' ' else None,
                encoding=encoding,
                skiprows=skip_rows,
                header=0
            )
        else:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter if delimiter != ' ' else None,
                encoding=encoding,
                skiprows=skip_rows,
                header=None
            )
            df.columns = [f"col_{c}" for c in range(df.shape[1])]
    except Exception as e:
        df = pd.read_csv(file_path, header=None)
        df.columns = [f"col_{c}" for c in range(df.shape[1])]
        
    # Clean trailing empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = sanitize_dataframe(df)
    
    # Column classification
    columns = list(df.columns)
    timestamp_col = None
    signal_col = None
    id_col = None
    
    # 1. Detect ID column
    for col in columns:
        col_lower = str(col).lower()
        if 'id' in col_lower or 'subject' in col_lower or 'name' in col_lower:
            id_col = col
            break
            
    # 2. Detect timestamp column
    for col in columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['time', 'timestamp', 'min', 'sec', 'ms', 't']):
            timestamp_col = col
            break
    # Fallback timestamp: check if strictly increasing
    if timestamp_col is None:
        for col in columns:
            if col == id_col:
                continue
            vals = df[col].to_numpy()
            if len(vals) > 1 and np.all(np.diff(vals) > 0):
                timestamp_col = col
                break
                
    # 3. Detect signal column
    if signal_col_override and signal_col_override in df.columns:
        signal_col = signal_col_override
    else:
        # Search for keyword matches
        for col in columns:
            if col in [timestamp_col, id_col]:
                continue
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['ppg', 'pulse', 'pleth', 'ir', 'red', 'ch3', 'channel 3', 'signal']):
                signal_col = col
                break
        # Fallback signal: last column that is not time/id
        if signal_col is None:
            remaining = [c for c in columns if c not in [timestamp_col, id_col]]
            if remaining:
                signal_col = remaining[-1]
            else:
                signal_col = columns[-1]
                
    file_info = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "detected_encoding": encoding,
        "detected_delimiter": delimiter,
        "data_start_line": data_start_idx + 1,
        "columns": columns,
        "timestamp_col": timestamp_col,
        "signal_col": signal_col,
        "id_col": id_col
    }
    
    return df, metadata, file_info
