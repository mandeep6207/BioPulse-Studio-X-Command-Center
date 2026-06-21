import os
import pandas as pd
import numpy as np
import scipy.signal
from typing import Dict, Any, List, Tuple
from biopulse_backend.core.input_engine import scan_and_classify_input
from biopulse_backend.core.file_loader import load_ppg_file
from biopulse_backend.core.signal_classifier import classify_signal
from biopulse_backend.core.fs_detector import detect_sampling_frequency
from biopulse_backend.core.orientation_detector import detect_orientation
from biopulse_backend.core.quality_engine import assess_signal_quality
from biopulse_backend.core.filter_playground import evaluate_filters
from biopulse_backend.core.preprocessor import preprocess_signal
from biopulse_backend.core.morphology_guard import guard_filter_morphology
from biopulse_backend.core.feature_validator import validate_features
from biopulse_backend.core.visualization_data import prepare_visualization_data
from biopulse_backend.core.download_center import (
    export_raw_csv,
    export_preprocessed_csv,
    export_features_csv,
    export_report_json,
    create_research_package,
    export_pdf_report,
    export_png_plot
)
from biopulse_backend.core.verification_engine import verify_signal_pipeline

def process_batch(input_path: str, output_dir: str) -> str:
    """
    Given a folder or ZIP of files, executes the full pipeline for each file.
    Aggregates results in output_dir/batch_summary.csv and packages them in output_dir/research_package.zip.
    Returns:
        path to research_package.zip
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scan and classify files
    classifications = scan_and_classify_input(input_path)
    
    batch_records = []
    generated_files = []
    
    for clf in classifications:
        file_name = os.path.basename(clf.file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Default fallback record if file is completely unsafe/fails loading
        record = {
            "file_name": file_name,
            "status": clf.status,
            "sampling_rate_hz": 0.0,
            "duration_seconds": 0.0,
            "quality_score": 0.0,
            "quality_band": "Poor",
            "readiness_score": 0.0,
            "readiness_category": "D",
            "overall_verdict": "Verification FAIL",
            "wording": "Not ready for biomedical research",
            "failure_reasons": clf.reason
        }
        
        if clf.status == "Unsafe":
            batch_records.append(record)
            continue
            
        try:
            # Stage 1: Load file
            df, metadata, file_info = load_ppg_file(clf.file_path)
            raw_sig = df[file_info["signal_col"]].to_numpy()
            
            # Interpolated clean vector for calculations
            # (Pandas Loader sanitizes and interpolates NaNs columns automatically)
            clean_sig = raw_sig
            
            # Stage 3: FS Detector
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
            
            # Stage 9: Morphology Guard (applies the best filter and reduces strength if needed)
            filtered_sig, degraded, guard_details = guard_filter_morphology(clean_sig, fs, best_filter)
            
            # Stage 10: Feature Validator
            readiness_score, category, features, report = validate_features(filtered_sig, fs, fs_status)
            
            # Stage 11: Visualization Prep
            vis_data = prepare_visualization_data(raw_sig, filtered_sig, fs, np.array(vis_peaks_find(filtered_sig, fs)), quality_metrics, features)
            
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
            
            # Get peak indices for PNG export
            peaks_indices = np.array(vis_data["peaks"])
            
            # Export individual files (Stage 12)
            raw_csv = export_raw_csv(df, os.path.join(output_dir, f"{base_name}_raw.csv"))
            prep_csv = export_preprocessed_csv(
                np.array(vis_data["signal_data"]["time"]),
                filtered_sig,
                np.array(vis_data["derivatives"]["vpg"]),
                np.array(vis_data["derivatives"]["apg"]),
                os.path.join(output_dir, f"{base_name}_preprocessed.csv")
            )
            feat_csv = export_features_csv(features, os.path.join(output_dir, f"{base_name}_features.csv"))
            
            # Full JSON Report
            report_dict = {
                "file_info": file_info,
                "classification": {
                    "duration_bucket": duration_bucket,
                    "source_guess": source_guess,
                    "duration_seconds": duration_sec
                },
                "sampling_rate": fs_details,
                "orientation": {
                    "detected": orientation,
                    "recommend_reverse": recommend_reverse,
                    "confidence": orient_confidence
                },
                "quality": {
                    "score": quality_score,
                    "band": quality_band,
                    "metrics": quality_metrics
                },
                "morphology_guard": guard_details,
                "features": features,
                "validation": report,
                "verification": {
                    "overall_verdict": verdict.overall_verdict,
                    "wording": verdict.wording,
                    "reasons": verdict.failure_reasons
                }
            }
            report_json = export_report_json(report_dict, os.path.join(output_dir, f"{base_name}_report.json"))
            
            # Export PNG plot
            prep_png = export_png_plot(
                np.array(vis_data["signal_data"]["time"]),
                raw_sig,
                filtered_sig,
                peaks_indices,
                fs,
                os.path.join(output_dir, f"{base_name}_waveform.png")
            )
            
            # Export PDF report
            res_pdf = {
                "file_info": file_info,
                "fs": fs,
                "duration_sec": duration_sec,
                "raw_sig": raw_sig,
                "filtered_sig": filtered_sig,
                "quality_score": quality_score,
                "quality_band": quality_band,
                "readiness_score": readiness_score,
                "category": category,
                "verdict": verdict,
                "quality_metrics": quality_metrics
            }
            pdf_report = export_pdf_report(
                res_pdf,
                prep_png,
                os.path.join(output_dir, f"{base_name}_report.pdf")
            )
            
            generated_files.extend([raw_csv, prep_csv, feat_csv, report_json, prep_png, pdf_report])
            
            # Update Record
            record = {
                "file_name": file_name,
                "status": clf.status,
                "sampling_rate_hz": fs,
                "duration_seconds": duration_sec,
                "quality_score": quality_score,
                "quality_band": quality_band,
                "readiness_score": readiness_score,
                "readiness_category": category,
                "overall_verdict": verdict.overall_verdict,
                "wording": verdict.wording,
                "failure_reasons": "; ".join(verdict.failure_reasons) if verdict.failure_reasons else "None"
            }
            
        except Exception as e:
            record["failure_reasons"] = f"Pipeline execution failed: {str(e)}"
            
        batch_records.append(record)
        
    # Write batch summary CSV
    summary_df = pd.DataFrame(batch_records)
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    generated_files.append(summary_path)
    
    # Create research package ZIP
    zip_path = os.path.join(output_dir, "research_package.zip")
    create_research_package(generated_files, zip_path, manifest_info={"total_files_processed": len(batch_records)})
    
    return zip_path

def vis_peaks_find(sig: np.ndarray, fs: float) -> List[int]:
    """Helper to find peak indices for vis data prep."""
    sig_detrend = scipy.signal.detrend(sig)
    dist = max(5, int(fs * 0.4))
    peaks, _ = scipy.signal.find_peaks(sig_detrend, distance=dist, prominence=0.1 * np.std(sig_detrend))
    return peaks.tolist()
