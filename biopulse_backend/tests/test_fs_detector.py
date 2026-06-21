import os
import pytest
import numpy as np
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.fs_detector import detect_sampling_frequency

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "clg1": os.path.join(SAMPLES_DIR, "Clg1_20250609_160754_1772.txt"),
        "clg2": os.path.join(SAMPLES_DIR, "Clg2_20250609_161257_5779.txt"),
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_3_fs_detector(sample_paths):
    print("\n=== RUNNING STAGE 3: FS DETECTOR TEST ===")
    
    # 1. Test Sub2
    df3, meta3, info3 = load_ppg_file(sample_paths["sub2"])
    sig3 = df3[info3["signal_col"]].to_numpy()
    fs3, conf3, status3, details3 = detect_sampling_frequency(sig3, info3, meta3, df3)
    
    print(f"Sub2 (Clinical CSV with timestamp/metadata):")
    print(f"  Detected Fs: {fs3} Hz")
    print(f"  Confidence: {conf3}")
    print(f"  Status: {status3}")
    print(f"  Source: {details3['source']}")
    print(f"  Estimated Fs from data: {details3['estimated_fs']} Hz")
    print(f"  Agreement score: {details3['agreement_score']}")
    print(f"  Votes: {details3['votes']}")
    
    assert fs3 == 1000.0
    assert status3 in ["PASS", "WARNING"]
    
    # 2. Test Clg1
    df1, meta1, info1 = load_ppg_file(sample_paths["clg1"])
    sig1 = df1[info1["signal_col"]].to_numpy()
    fs1, conf1, status1, details1 = detect_sampling_frequency(sig1, info1, meta1, df1)
    
    print(f"Clg1 (Mobile PPG - pure signal values):")
    print(f"  Detected Fs: {fs1} Hz")
    print(f"  Confidence: {conf1}")
    print(f"  Status: {status1}")
    print(f"  Source: {details1['source']}")
    print(f"  Agreement score: {details1['agreement_score']}")
    print(f"  Votes: {details1['votes']}")
    
    # 3. Test Clg2
    df2, meta2, info2 = load_ppg_file(sample_paths["clg2"])
    sig2 = df2[info2["signal_col"]].to_numpy()
    fs2, conf2, status2, details2 = detect_sampling_frequency(sig2, info2, meta2, df2)
    
    print(f"Clg2 (Mobile PPG - pure signal values):")
    print(f"  Detected Fs: {fs2} Hz")
    print(f"  Confidence: {conf2}")
    print(f"  Status: {status2}")
    print(f"  Source: {details2['source']}")
    print(f"  Agreement score: {details2['agreement_score']}")
    print(f"  Votes: {details2['votes']}")
