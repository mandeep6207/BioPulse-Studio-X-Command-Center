import pytest
from biopulse_backend.core.verification_engine import verify_signal_pipeline

def test_stage_16_verification_engine():
    print("\n=== RUNNING STAGE 16: VERIFICATION ENGINE TEST ===")
    
    # 1. Normal PASS case
    res_pass = verify_signal_pipeline(
        file_name="test_clean.csv",
        file_loader_success=True,
        fs_status="PASS",
        fs_val=250.0,
        preprocess_success=True,
        morphology_guard_status="PASSED",
        feature_category="A",
        readiness_score=100.0,
        quality_score=95.0,
        quality_band="Excellent"
    )
    
    print(f"PASS case wording: '{res_pass.wording}'")
    assert res_pass.overall_verdict == "Verification PASS"
    assert "Ready for biomedical research" in res_pass.wording
    assert not res_pass.failure_reasons
    
    # 2. FAIL case: FS unresolved
    res_fail_fs = verify_signal_pipeline(
        file_name="test_missing_fs.csv",
        file_loader_success=True,
        fs_status="FS_UNRESOLVED",
        fs_val=100.0,
        preprocess_success=True,
        morphology_guard_status="PASSED",
        feature_category="B",
        readiness_score=85.0,
        quality_score=80.0,
        quality_band="Good"
    )
    
    print(f"FAIL case (FS unresolved) wording: '{res_fail_fs.wording}'")
    print(f"  Reasons: {res_fail_fs.failure_reasons}")
    assert res_fail_fs.overall_verdict == "Verification FAIL"
    assert "Not ready for biomedical research" in res_fail_fs.wording
    assert len(res_fail_fs.failure_reasons) == 1
    
    # 3. FAIL case: Multi-module failure
    res_fail_multi = verify_signal_pipeline(
        file_name="test_corrupt.csv",
        file_loader_success=False,
        fs_status="FS_UNRESOLVED",
        fs_val=100.0,
        preprocess_success=False,
        morphology_guard_status="DEGRADED_FALLBACK",
        feature_category="D",
        readiness_score=30.0,
        quality_score=10.0,
        quality_band="Poor"
    )
    
    print(f"FAIL case (multi-failure) wording: '{res_fail_multi.wording}'")
    print(f"  Reasons: {res_fail_multi.failure_reasons}")
    assert res_fail_multi.overall_verdict == "Verification FAIL"
    assert len(res_fail_multi.failure_reasons) == 5
