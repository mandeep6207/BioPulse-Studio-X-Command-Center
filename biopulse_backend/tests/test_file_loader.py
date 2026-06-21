import os
import pytest
from biopulse_backend.core.input_engine import scan_and_classify_input
from biopulse_backend.core.file_loader import load_ppg_file

# Paths to test samples
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "clg1": os.path.join(SAMPLES_DIR, "Clg1_20250609_160754_1772.txt"),
        "clg2": os.path.join(SAMPLES_DIR, "Clg2_20250609_161257_5779.txt"),
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_0_input_engine(sample_paths):
    print("\n=== RUNNING STAGE 0: INPUT ENGINE TEST ===")
    for key, path in sample_paths.items():
        assert os.path.exists(path), f"Sample file not found at {path}"
        classifications = scan_and_classify_input(path)
        assert len(classifications) == 1
        clf = classifications[0]
        print(f"File: {clf.file_path}")
        print(f"  Status: {clf.status}")
        print(f"  Reason: {clf.reason}")
        print(f"  Size: {clf.file_size_bytes} bytes")
        print(f"  Extension: {clf.file_extension}")
        assert clf.status in ["Ready", "Partially Ready", "Research Only"]

def test_stage_1_file_loader(sample_paths):
    print("\n=== RUNNING STAGE 1: FILE LOADER TEST ===")
    
    # 1. Test Clg1
    df1, meta1, info1 = load_ppg_file(sample_paths["clg1"])
    print(f"Clg1 Loaded:")
    print(f"  Columns: {info1['columns']}")
    print(f"  Signal Column: {info1['signal_col']}")
    print(f"  Timestamp Column: {info1['timestamp_col']}")
    print(f"  Shape: {df1.shape}")
    assert df1.shape[0] == 2400
    assert info1["signal_col"] is not None
    
    # 2. Test Clg2
    df2, meta2, info2 = load_ppg_file(sample_paths["clg2"])
    print(f"Clg2 Loaded:")
    print(f"  Columns: {info2['columns']}")
    print(f"  Signal Column: {info2['signal_col']}")
    print(f"  Timestamp Column: {info2['timestamp_col']}")
    print(f"  Shape: {df2.shape}")
    assert df2.shape[0] == 2400
    assert info2["signal_col"] is not None
    
    # 3. Test Sub2
    df3, meta3, info3 = load_ppg_file(sample_paths["sub2"])
    print(f"Sub2 Loaded:")
    print(f"  Columns: {info3['columns']}")
    print(f"  Signal Column: {info3['signal_col']}")
    print(f"  Timestamp Column: {info3['timestamp_col']}")
    print(f"  Shape: {df3.shape}")
    print(f"  Metadata Key-Values:")
    for k, v in list(meta3.items())[:5]:
        print(f"    {k}: {v}")
    
    # Assertions for Sub2.csv
    assert df3.shape[0] > 0
    assert info3["timestamp_col"] == "min"
    assert info3["signal_col"] == "CH3"
