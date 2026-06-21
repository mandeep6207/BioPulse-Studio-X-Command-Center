# BioPulse Studio X — Phase 1 Backend Pipeline

A photoplethysmography (PPG) signal processing and validation backend library for **biomedical research** (not for clinical diagnosis).

This package implements the frozen 17-stage signal processing architecture, providing robust, deterministic, and unit-tested functions for file loading, sampling rate estimation, signal conditioning, feature extraction, and pipeline verification.

---

## Installation & Requirements

Ensure you have Python 3.9+ installed. The dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Stage-by-Stage Usage

### Stage 0 — Input Engine (`biopulse_backend.core.input_engine`)
Scans files, folders, or ZIP archives and evaluates input safety classifications (`Ready`, `Partially Ready`, `Research Only`, `Unsafe`).

```python
from biopulse_backend.core.input_engine import scan_and_classify_input

classifications = scan_and_classify_input("data/test_samples/Sub2.csv")
for clf in classifications:
    print(f"File: {clf.file_path} | Status: {clf.status} | Reason: {clf.reason}")
```

### Stage 1 — Universal File Loader (`biopulse_backend.core.file_loader`)
Auto-decodes encodings, detects delimiters, extracts metadata headers, and classifies column layouts (signal, timestamp, and ID).

```python
from biopulse_backend.core.file_loader import load_ppg_file

df, metadata, file_info = load_ppg_file("data/test_samples/Sub2.csv")
print("Signal column detected:", file_info["signal_col"])
```

### Stage 2 — Signal Classifier (`biopulse_backend.core.signal_classifier`)
Categorizes files into duration buckets (`<20s Short`, `20-120s Standard`, etc.) and infers the signal source (`Mobile`, `Clinical`, `Research`, `Unknown`).

```python
from biopulse_backend.core.signal_classifier import classify_signal

duration_bucket, source_guess, duration_sec = classify_signal(signal, fs, file_info, metadata)
```

### Stage 3 — FS Detector (`biopulse_backend.core.fs_detector`)
Identifies the sampling frequency ($Fs$) via a priority chain (User Override $\rightarrow$ Metadata $\rightarrow$ Timestamp $\rightarrow$ Filename $\rightarrow$ Voting). Validates it using FFT, Autocorrelation, and PSD.

```python
from biopulse_backend.core.fs_detector import detect_sampling_frequency

fs, confidence, status, details = detect_sampling_frequency(signal, file_info, metadata, df)
```

### Stage 4 — User Controls (`biopulse_backend.core.user_controls`)
Applies overrides, reversing, trimming, centering (Mean/Median/Detrend), and temporal chunking.

```python
from biopulse_backend.core.user_controls import apply_centering, apply_trim

centered = apply_centering(signal, method="detrend")
trimmed_sig, _ = apply_trim(centered, fs, trim_start_sec=1.0, trim_end_sec=1.0)
```

### Stage 5 — Orientation Detector (`biopulse_backend.core.orientation_detector`)
Uses skewness and peak prominence ratios to identify if the signal is inverted, returning orientation (`normal`, `inverted`, `uncertain`) and a reverse recommendation flag.

```python
from biopulse_backend.core.orientation_detector import detect_orientation

orientation, recommend_reverse, confidence = detect_orientation(signal, fs)
```

### Stage 6 — Quality Engine (`biopulse_backend.core.quality_engine`)
Computes detailed metrics (SNR, drift ratio, flatline percentage, entropy, peak stability, and beat consistency) and maps them to a quality score and band (`Excellent`, `Very Good`, `Good`, `Acceptable`, `Poor`).

```python
from biopulse_backend.core.quality_engine import assess_signal_quality

score, band, metrics = assess_signal_quality(raw_signal, clean_signal, fs)
```

### Stage 7 — Filter Playground (`biopulse_backend.core.filter_playground`)
Evaluates 7 filters (Butterworth, Chebyshev, Savitzky-Golay, Gaussian, Median, Wavelet, and None) against morphology retention, peak retention, beat similarity, and SNR.

```python
from biopulse_backend.core.filter_playground import evaluate_filters

best_filter, needs_comparison, scoreboard, filtered_signals = evaluate_filters(signal, fs)
```

### Stage 8 — Preprocessor (`biopulse_backend.core.preprocessor`)
Applies exactly one filter and verifies that the dicrotic notch is preserved using APG derivative wiggles.

```python
from biopulse_backend.core.preprocessor import preprocess_signal

filtered_sig, notch_preserved, details = preprocess_signal(signal, fs, "Butterworth")
```

### Stage 9 — Morphology Guard (`biopulse_backend.core.morphology_guard`)
Compares pre/post characteristics (HR, RR, pulse width, amplitude) and automatically reduces filter strength (up to 3 retries) if deviations exceed thresholds.

```python
from biopulse_backend.core.morphology_guard import guard_filter_morphology

guarded_sig, degraded, details = guard_filter_morphology(signal, fs, "Butterworth")
```

### Stage 10 — Feature Validator (`biopulse_backend.core.feature_validator`)
Extracts features (energy, area, entropy, Poincare SD1/SD2, VPG/APG derivatives) and runs physiological checks. Caps readiness if FS is unresolved.

```python
from biopulse_backend.core.feature_validator import validate_features

readiness_score, category, features, report = validate_features(filtered_sig, fs, fs_status)
```

### Stage 11 — Visualization Data Prep (`biopulse_backend.core.visualization_data`)
Structures data vectors, peak indices, derivatives, trends, radar plot coordinates, and chunk-wise noise matrices ready for Plotly.

```python
from biopulse_backend.core.visualization_data import prepare_visualization_data

vis_data = prepare_visualization_data(raw_sig, filtered_sig, fs, peaks, quality_metrics, features)
```

### Stage 12 — Download Center (`biopulse_backend.core.download_center`)
Exports Raw, Preprocessed, Features, and JSON reports, and packages them into a research ZIP with a manifest.

```python
from biopulse_backend.core.download_center import create_research_package

zip_path = create_research_package([raw_csv, prep_csv, feat_csv, json_report], "output/package.zip")
```

### Stage 13 — Batch Processor (`biopulse_backend.core.batch_processor`)
Processes a folder or ZIP file, executing the full pipeline for each file, and aggregates reports.

```python
from biopulse_backend.core.batch_processor import process_batch

zip_path = process_batch("data/test_samples", "data/output_batch")
```

### Stage 16 — Verification Engine (`biopulse_backend.core.verification_engine`)
Validates that all critical stages succeeded and issues a verdict: `Verification PASS` or `Verification FAIL`.

```python
from biopulse_backend.core.verification_engine import verify_signal_pipeline

verdict = verify_signal_pipeline(file_name, loader_success, fs_status, fs, prep_success, guard_status, category, readiness_score, quality_score, quality_band)
print(verdict.overall_verdict) # e.g. "Verification PASS"
print(verdict.wording)         # "Ready for biomedical research"
```

---

## Running the UI Dashboard

Start the Biomedical Command Center Streamlit application:

```bash
streamlit run biopulse_backend/app.py
```

---

## Running Tests

Run all unit tests, E2E tests, and UI data binding validation:

```bash
python -m pytest -s biopulse_backend/tests/
```
