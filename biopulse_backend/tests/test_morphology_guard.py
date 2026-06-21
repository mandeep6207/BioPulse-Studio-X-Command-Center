import os
import pytest
import numpy as np
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.morphology_guard import guard_filter_morphology

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_9_morphology_guard(sample_paths):
    print("\n=== RUNNING STAGE 9: MORPHOLOGY GUARD TEST ===")
    
    df, meta, info = load_ppg_file(sample_paths["sub2"])
    sig = df[info["signal_col"]].to_numpy()
    fs, _, _, _ = detect_sampling_frequency(sig, info, meta, df)
    
    # Run morphology guard with Butterworth filter
    guarded_sig, degraded, details = guard_filter_morphology(sig, fs, "Butterworth")
    
    print("Morphology Guard Details:")
    print(f"  Degraded: {degraded}")
    print(f"  Final Strength Attempt: {details['final_strength_attempt']}")
    print(f"  Status: {details['status']}")
    
    for attempt in ["attempt_0", "attempt_1", "attempt_2"]:
        if attempt in details:
            att = details[attempt]
            print(f"  {attempt}: passed={att['passed']} | HR dev={att['hr_deviation']:.4f} | PeakCount dev={att['peak_count_deviation']:.4f} | Amp dev={att['amplitude_deviation']:.4f}")
            
    assert len(guarded_sig) == len(sig)
    assert isinstance(degraded, bool)
