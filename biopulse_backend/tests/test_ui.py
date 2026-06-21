import os
import pytest
from biopulse_backend.app import process_file_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "test_samples")

@pytest.fixture
def sample_paths():
    return {
        "clg1": os.path.join(SAMPLES_DIR, "Clg1_20250609_160754_1772.txt"),
        "clg2": os.path.join(SAMPLES_DIR, "Clg2_20250609_161257_5779.txt"),
        "sub2": os.path.join(SAMPLES_DIR, "Sub2.csv")
    }

def test_ui_data_binding_pipeline(sample_paths):
    print("\n=== RUNNING PROGRAMMATIC UI INTEGRATION TEST ===")
    
    for name, path in sample_paths.items():
        assert os.path.exists(path)
        print(f"Testing UI data model for {os.path.basename(path)}...")
        
        # Run the pipeline function invoked by Streamlit
        res = process_file_pipeline(path)
        
        # Verify returned data keys used by Plotly graphs and KPI metrics
        assert "vis_data" in res
        assert "signal_data" in res["vis_data"]
        assert "time" in res["vis_data"]["signal_data"]
        assert "raw" in res["vis_data"]["signal_data"]
        assert "preprocessed" in res["vis_data"]["signal_data"]
        
        assert "peaks" in res["vis_data"]
        assert "derivatives" in res["vis_data"]
        assert "vpg" in res["vis_data"]["derivatives"]
        assert "apg" in res["vis_data"]["derivatives"]
        
        assert "radar" in res["vis_data"]
        assert "quality" in res["vis_data"]["radar"]
        assert "morphology" in res["vis_data"]["radar"]
        
        assert "heatmap" in res["vis_data"]
        assert "matrix" in res["vis_data"]["heatmap"]
        
        # Verify KPI values are populated
        assert res["fs"] > 0.0
        assert res["duration_sec"] > 0.0
        assert res["quality_score"] >= 0.0
        assert res["readiness_score"] >= 0.0
        assert res["verdict"] is not None
        
        print(f"  OK: All Plotly layout bindings and KPI outputs present for {name}.")
