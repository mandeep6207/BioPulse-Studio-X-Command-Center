import os
import pytest
import numpy as np
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.quality_engine import assess_signal_quality

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "clg1": os.path.join(SAMPLES_DIR, "Clg1_20250609_160754_1772.txt"),
        "clg2": os.path.join(SAMPLES_DIR, "Clg2_20250609_161257_5779.txt"),
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_6_quality_engine(sample_paths):
    print("\n=== RUNNING STAGE 6: QUALITY ENGINE TEST ===")
    
    for name, path in sample_paths.items():
        df, meta, info = load_ppg_file(path)
        sig = df[info["signal_col"]].to_numpy()
        
        # Detect FS first
        fs, _, _, _ = detect_sampling_frequency(sig, info, meta, df)
        
        # Assess quality (using sig as both raw and interpolated for this test)
        score, band, metrics = assess_signal_quality(sig, sig, fs)
        
        print(f"\nFile: {info['file_name']} (Fs = {fs} Hz)")
        print(f"  Quality Score: {score:.2f} / 100")
        print(f"  Band: {band}")
        print(f"  Metrics Breakdown:")
        print(f"    SNR: {metrics['snr_db']:.2f} dB")
        print(f"    Baseline Drift Std: {metrics['baseline_wander_std']:.5f}")
        print(f"    Drift Ratio: {metrics['drift_ratio']:.4f}")
        print(f"    Motion Artifacts: {metrics['motion_artifact_pct']:.2f}%")
        print(f"    Shannon Entropy: {metrics['entropy']:.3f}")
        print(f"    Flat Regions: {metrics['flat_region_pct']:.2f}%")
        print(f"    Missing Data: {metrics['missing_data_pct']:.2f}%")
        print(f"    Kurtosis: {metrics['kurtosis']:.2f}")
        print(f"    Skewness: {metrics['skewness']:.2f}")
        print(f"    Peak Stability (CV of prominence): {metrics['peak_stability']:.3f}")
        print(f"    Beat Consistency: {metrics['beat_consistency']:.3f}")
        print(f"    Detected Peaks Count: {metrics['peak_count']}")
        
        assert 0.0 <= score <= 100.0
        assert band in ["Excellent", "Very Good", "Good", "Acceptable", "Poor"]
