import os
import pytest
import pandas as pd
from biopulse_backend.core.batch_processor import process_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output_batch")

def test_stage_13_batch_processor():
    print("\n=== RUNNING STAGE 13: BATCH PROCESSOR TEST ===")
    
    zip_path = process_batch(SAMPLES_DIR, OUTPUT_DIR)
    
    print(f"Batch processing completed.")
    print(f"  Research package saved: {zip_path}")
    assert os.path.exists(zip_path)
    
    summary_csv = os.path.join(OUTPUT_DIR, "batch_summary.csv")
    print(f"  Summary CSV saved: {summary_csv}")
    assert os.path.exists(summary_csv)
    
    # Read summary and print it
    df = pd.read_csv(summary_csv)
    print("\nBatch Summary Table:")
    print(df[["file_name", "sampling_rate_hz", "quality_band", "readiness_category", "overall_verdict"]])
    
    # Verify row count is equal to the number of input files
    # The folder has Clg1, Clg2, and Sub2.csv (3 files)
    assert df.shape[0] == 3
