import os
import json
import zipfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def export_raw_csv(df: pd.DataFrame, file_path: str) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path

def export_preprocessed_csv(
    time_array: np.ndarray,
    preprocessed_signal: np.ndarray,
    vpg: np.ndarray,
    apg: np.ndarray,
    file_path: str
) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df = pd.DataFrame({
        "time_sec": time_array,
        "preprocessed_signal": preprocessed_signal,
        "vpg_1st_derivative": vpg,
        "apg_2nd_derivative": apg
    })
    df.to_csv(file_path, index=False)
    return file_path

def export_features_csv(features: Dict[str, Any], file_path: str) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert dict to a 1-row DataFrame
    df = pd.DataFrame([features])
    df.to_csv(file_path, index=False)
    return file_path

def export_report_json(report_dict: Dict[str, Any], file_path: str) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Handle numpy arrays in serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, cls=NumpyEncoder, indent=4)
    return file_path

def create_research_package(
    export_files: List[str], 
    zip_path: str,
    manifest_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Creates a ZIP packaging containing all generated CSV/JSON reports,
    plus an automatically compiled manifest.json.
    """
    zip_path = os.path.abspath(zip_path)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    # Create manifest
    manifest = {
        "package_type": "BioPulse Research Package",
        "verdict_wording": "Ready for biomedical research",
        "files_included": [os.path.basename(f) for f in export_files]
    }
    if manifest_info:
        manifest.update(manifest_info)
        
    temp_manifest_path = os.path.join(os.path.dirname(zip_path), "manifest.json")
    with open(temp_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    all_files = export_files + [temp_manifest_path]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in all_files:
            if os.path.exists(file):
                zip_file.write(file, os.path.basename(file))
                
    # Clean up temp manifest file
    if os.path.exists(temp_manifest_path):
        os.remove(temp_manifest_path)
        
    return zip_path

def export_pdf_report(report_dict: Dict[str, Any], file_path: str) -> None:
    """
    Stub for PDF report generation - deferred to Phase 2.
    """
    raise NotImplementedError("PDF/PNG report rendering is deferred to Phase 2 UI Integration.")
