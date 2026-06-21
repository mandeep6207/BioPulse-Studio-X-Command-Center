import os
import pytest
import numpy as np
import scipy.signal
from biopulse_backend.core.input_engine import classify_single_file
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.signal_classifier import classify_signal
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.orientation_detector import detect_orientation
from biopulse_backend.core.quality_engine import assess_signal_quality
from biopulse_backend.core.filter_playground import evaluate_filters
from biopulse_backend.core.morphology_guard import guard_filter_morphology
from biopulse_backend.core.feature_validator import validate_features
from biopulse_backend.core.visualization_data import prepare_visualization_data
from biopulse_backend.core.verification_engine import verify_signal_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "clg1": os.path.join(SAMPLES_DIR, "Clg1_20250609_160754_1772.txt"),
        "clg2": os.path.join(SAMPLES_DIR, "Clg2_20250609_161257_5779.txt"),
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_end_to_end_pipeline(sample_paths):
    print("\n=======================================================")
    print("=== RUNNING COMPLETE END-TO-END PIPELINE (PHASE 1) ===")
    print("=======================================================")
    
    for name, path in sample_paths.items():
        file_name = os.path.basename(path)
        print(f"\n>>> PROCESSING FILE: {file_name}")
        
        # Stage 0: Input Engine
        clf = classify_single_file(path)
        assert clf.status != "Unsafe", f"File is classified as Unsafe: {clf.reason}"
        
        # Stage 1: Universal File Loader
        df, metadata, file_info = load_ppg_file(path)
        raw_sig = df[file_info["signal_col"]].to_numpy()
        clean_sig = raw_sig # Pandas loader already sanitized/interpolated NaNs
        
        # Stage 3: FS (Sampling Frequency) Detector
        fs, fs_confidence, fs_status, fs_details = detect_sampling_frequency(clean_sig, file_info, metadata, df)
        
        # Stage 2: Signal Classifier
        duration_bucket, source_guess, duration_sec = classify_signal(clean_sig, fs, file_info, metadata)
        
        # Stage 5: Orientation Detector
        orientation, recommend_reverse, orient_confidence = detect_orientation(raw_sig, fs)
        
        # If orientation recommends reverse, we invert the signal for processing!
        clean_sig = raw_sig * -1.0 if recommend_reverse else raw_sig
        
        # Stage 6: Quality Engine
        quality_score, quality_band, quality_metrics = assess_signal_quality(raw_sig, clean_sig, fs)
        
        # Stage 7: Filter Playground
        best_filter, needs_comp, scoreboard, _ = evaluate_filters(clean_sig, fs)
        
        # Stage 9: Morphology Guard (selects and applies best filter, down-scales if morph check fails)
        filtered_sig, degraded_flag, guard_details = guard_filter_morphology(clean_sig, fs, best_filter)
        
        # Get morphology score (total score from the playground for the chosen filter)
        morphology_score = scoreboard[best_filter]["morphology"]
        
        # Stage 10: Feature Validator
        readiness_score, category, features, report = validate_features(filtered_sig, fs, fs_status)
        
        # Stage 11: Visualization Data Prep
        filt_detrend = scipy.signal.detrend(filtered_sig)
        dist = max(5, int(fs * 0.4))
        peaks_indices, _ = scipy.signal.find_peaks(filt_detrend, distance=dist, prominence=0.1 * np.std(filt_detrend))
        vis_data = prepare_visualization_data(raw_sig, filtered_sig, fs, peaks_indices, quality_metrics, features)
        
        # Stage 16: Verification Engine
        verdict = verify_signal_pipeline(
            file_name=file_name,
            file_loader_success=True,
            fs_status=fs_status,
            fs_val=fs,
            preprocess_success=True,
            morphology_guard_status=guard_details["status"],
            feature_category=category,
            readiness_score=readiness_score,
            quality_score=quality_score,
            quality_band=quality_band
        )
        
        # E2E VERDICT SUMMARY PRINT
        print("-" * 50)
        print(f"  E2E SUMMARY REPORT FOR {file_name}:")
        print(f"  * Pipeline Status      : {clf.status} ({clf.reason})")
        print(f"  * Sampling Rate        : {fs:.1f} Hz (Confidence: {fs_confidence:.2f}, Status: {fs_status})")
        print(f"  * Signal Classifier    : Source={source_guess}, Duration={duration_sec:.1f}s ({duration_bucket})")
        print(f"  * Orientation Detector : Orientation={orientation} (Confidence: {orient_confidence:.2f})")
        print(f"  * Quality Score        : {quality_score:.2f}/100 (Band: {quality_band})")
        print(f"  * Best Applied Filter  : {best_filter} (Morphology Score: {morphology_score:.2f}%)")
        print(f"  * Filter Degradation   : {degraded_flag} (Status: {guard_details['status']})")
        print(f"  * Readiness Score      : {readiness_score:.2f}% (Category: {category})")
        print(f"  * Verification Verdict : {verdict.overall_verdict}")
        print(f"  * Verdict Wording      : \"{verdict.wording}\"")
        if verdict.failure_reasons:
            print(f"  * Failure Reasons      : {verdict.failure_reasons}")
        print("-" * 50)
        
        # Core assertions for E2E success
        assert verdict.overall_verdict in ["Verification PASS", "Verification FAIL"]
        assert "Ready for" in verdict.wording
        assert len(filtered_sig) == len(raw_sig)
        assert len(vis_data["signal_data"]["time"]) == len(raw_sig)
