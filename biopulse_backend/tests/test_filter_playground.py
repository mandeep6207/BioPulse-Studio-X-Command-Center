import os
import pytest
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.filter_playground import evaluate_filters

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "clg1": os.path.join(SAMPLES_DIR, "Clg1_20250609_160754_1772.txt"),
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_stage_7_filter_playground(sample_paths):
    print("\n=== RUNNING STAGE 7: FILTER PLAYGROUND TEST ===")
    
    # We test on Sub2.csv
    df, meta, info = load_ppg_file(sample_paths["sub2"])
    sig = df[info["signal_col"]].to_numpy()
    fs, _, _, _ = detect_sampling_frequency(sig, info, meta, df)
    
    best_filter, needs_comp, scoreboard, _ = evaluate_filters(sig, fs)
    
    print(f"\nScoreboard for {info['file_name']}:")
    print(f"  Best Auto-Picked Filter: {best_filter}")
    print(f"  Needs Manual Comparison: {needs_comp}")
    print(f"  Scores:")
    for f_name, metrics in scoreboard.items():
        print(f"    {f_name:15}: Total={metrics['total_score']:.2f} | Morph={metrics['morphology']:.2f} | PeakRet={metrics['peak_retention']:.2f} | BeatSim={metrics['beat_similarity']:.2f} | SNR_Score={metrics['snr_score']:.2f} | SNR={metrics['snr_db']:.2f} dB")
        
    assert best_filter in scoreboard
    assert len(scoreboard) == 7
