import os
import pytest
import numpy as np
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.preprocessor import preprocess_signal
from biopulse_backend.core.feature_validator import validate_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_10_feature_validator(sample_paths):
    print("\n=== RUNNING STAGE 10: FEATURE VALIDATOR TEST ===")
    
    df, meta, info = load_ppg_file(sample_paths["sub2"])
    sig = df[info["signal_col"]].to_numpy()
    fs, _, fs_status, _ = detect_sampling_frequency(sig, info, meta, df)
    
    # Preprocess
    filtered, _, _ = preprocess_signal(sig, fs, "Butterworth")
    
    # Validate features under normal FS_status (PASS)
    score, cat, features, report = validate_features(filtered, fs, fs_status)
    
    print("Normal FS Case:")
    print(f"  Readiness Score: {score:.2f}%")
    print(f"  Category: {cat}")
    print(f"  Passed Checks Count: {report['passed_count']} / {report['total_count']}")
    print(f"  Validated Features:")
    for k, v in list(features.items())[:6]:
        print(f"    {k}: {v:.4f}")
        
    assert 0.0 <= score <= 100.0
    assert cat in ["A", "B", "C", "D"]
    
    # Now simulate FS_UNRESOLVED to check capping
    cap_score, cap_cat, cap_features, cap_report = validate_features(filtered, fs, "FS_UNRESOLVED")
    
    print("\nFS_UNRESOLVED Simulated Case:")
    print(f"  Capped Readiness Score: {cap_score:.2f}%")
    print(f"  Capped Category: {cap_cat}")
    
    # Verify the cap
    assert cap_score <= 85.0
    assert cap_cat != "A"
