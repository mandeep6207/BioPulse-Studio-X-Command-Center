import os
import zipfile
import tempfile
import pathlib
from dataclasses import dataclass
from typing import List, Union, Optional
from biopulse_backend.utils.dataframe_utils import detect_encoding, detect_delimiter

@dataclass
class InputClassification:
    file_path: str
    status: str          # Ready / Partially Ready / Research Only / Unsafe
    reason: str
    file_size_bytes: int
    file_extension: str

def classify_single_file(file_path: str) -> InputClassification:
    """
    Classifies a single input file into Ready / Partially Ready / Research Only / Unsafe.
    """
    file_path = os.path.abspath(file_path)
    
    # 1. Existence check
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return InputClassification(
            file_path=file_path,
            status="Unsafe",
            reason="File does not exist or is not a file.",
            file_size_bytes=0,
            file_extension=""
        )
        
    size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    
    # 2. Empty check
    if size == 0:
        return InputClassification(
            file_path=file_path,
            status="Unsafe",
            reason="File is empty (0 bytes).",
            file_size_bytes=0,
            file_extension=ext
        )
        
    # 3. Readability & Encoding check
    enc = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=enc) as f:
            first_chars = f.read(2048)
    except Exception as e:
        return InputClassification(
            file_path=file_path,
            status="Unsafe",
            reason=f"File unreadable: Unicode decoding failed. Details: {str(e)}",
            file_size_bytes=size,
            file_extension=ext
        )
        
    # 4. Check for binary or corrupted non-text characters
    # If the file contains null bytes or a high proportion of non-ascii unprintables, classify as Unsafe.
    if '\x00' in first_chars:
        return InputClassification(
            file_path=file_path,
            status="Unsafe",
            reason="Binary data detected (null bytes present).",
            file_size_bytes=size,
            file_extension=ext
        )
        
    # 5. Extension and delimiter check
    if ext not in ['.csv', '.txt', '.tsv']:
        # We don't reject outright, but label as Research Only
        return InputClassification(
            file_path=file_path,
            status="Research Only",
            reason=f"Unrecognized file extension '{ext}'. Will attempt custom parsing.",
            file_size_bytes=size,
            file_extension=ext
        )
        
    delim = detect_delimiter(file_path, enc)
    
    # Check if lines have numeric values or if it's completely scrambled text
    lines = [l.strip() for l in first_chars.split('\n') if l.strip()]
    numeric_count = 0
    total_tokens = 0
    
    for line in lines[:10]:
        tokens = line.split(delim) if delim != ' ' else line.split()
        for tok in tokens:
            tok_clean = tok.strip().replace('"', '').replace("'", "")
            total_tokens += 1
            # Check if token is float-like
            try:
                float(tok_clean)
                numeric_count += 1
            except ValueError:
                # Might be header or label
                pass
                
    if total_tokens > 0 and (numeric_count / total_tokens) < 0.1:
        # Mostly non-numeric text and not a standard format
        return InputClassification(
            file_path=file_path,
            status="Research Only",
            reason="Low density of numeric data in preview. Possible high-text metadata file.",
            file_size_bytes=size,
            file_extension=ext
        )
        
    # 6. Success classification
    if ext == '.csv' and delim == ',':
        status = "Ready"
        reason = "Standard CSV with comma delimiter."
    elif ext == '.txt' and delim in ['\t', ' ', ',']:
        status = "Ready"
        reason = "Standard delimited TXT file."
    else:
        status = "Partially Ready"
        reason = f"Delimited data found with non-standard separator '{delim}' or format."
        
    return InputClassification(
        file_path=file_path,
        status=status,
        reason=reason,
        file_size_bytes=size,
        file_extension=ext
    )

def scan_and_classify_input(input_path: str, extract_dir: Optional[str] = None) -> List[InputClassification]:
    """
    Recursively scans input_path which can be:
    - A single file
    - A folder (scans recursively)
    - A ZIP file (extracts to extract_dir and scans contents)
    Returns a list of InputClassification objects.
    """
    input_path = os.path.abspath(input_path)
    classifications = []
    
    if not os.path.exists(input_path):
        # Return Unsafe directly
        return [InputClassification(
            file_path=input_path,
            status="Unsafe",
            reason="Input path does not exist.",
            file_size_bytes=0,
            file_extension=""
        )]
        
    # Check if ZIP file
    if os.path.isfile(input_path) and zipfile.is_zipfile(input_path):
        if extract_dir is None:
            # Create a temp dir inside the workspace so it conforms to guidelines
            extract_dir = os.path.join(os.path.dirname(input_path), "temp_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            # Scan the extracted directory
            return scan_and_classify_input(extract_dir, extract_dir)
        except Exception as e:
            return [InputClassification(
                file_path=input_path,
                status="Unsafe",
                reason=f"Failed to extract ZIP: {str(e)}",
                file_size_bytes=os.path.getsize(input_path),
                file_extension=".zip"
            )]
            
    # Check if single file
    if os.path.isfile(input_path):
        classifications.append(classify_single_file(input_path))
        
    # Check if folder
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            # Skip hidden files or folders
            for file in files:
                if file.startswith('.') or '__pycache__' in root:
                    continue
                file_full = os.path.join(root, file)
                classifications.append(classify_single_file(file_full))
                
    return classifications
