import os
import pytest
import numpy as np
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.preprocessor import preprocess_signal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_8_preprocessor(sample_paths):
    print("\n=== RUNNING STAGE 8: PREPROCESSOR TEST ===")
    
    df, meta, info = load_ppg_file(sample_paths["sub2"])
    sig = df[info["signal_col"]].to_numpy()
    fs, _, _, _ = detect_sampling_frequency(sig, info, meta, df)
    
    # Preprocess with Butterworth
    filtered_sig, notch_pres, details = preprocess_signal(sig, fs, "Butterworth")
    
    print(f"Preprocessor results for {info['file_name']}:")
    print(f"  Applied Filter: {details['applied_filter']}")
    print(f"  Notch Preserved: {notch_pres}")
    print(f"  Raw APG Peaks/Beat: {details['raw_apg_peaks_per_beat']:.3f}")
    print(f"  Filtered APG Peaks/Beat: {details['filtered_apg_peaks_per_beat']:.3f}")
    
    assert len(filtered_sig) == len(sig)
    assert details["applied_filter"] == "Butterworth"
    # For Butterworth bandpass 0.5-5.0 Hz at 1000 Hz, the notch is generally preserved or slightly smoothed.
    # We check that the return types and structure match.
    assert isinstance(notch_pres, bool)
