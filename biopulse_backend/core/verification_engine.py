from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class VerificationResult:
    file_name: str
    overall_verdict: str        # "Verification PASS" or "Verification FAIL"
    readiness_pct: float
    readiness_category: str     # A, B, C, D
    quality_score: float
    quality_band: str
    fs_hz: float
    wording: str               # "Ready for biomedical research" or "Not ready for biomedical research"
    failure_reasons: List[str]

def verify_signal_pipeline(
    file_name: str,
    file_loader_success: bool,
    fs_status: str,  # "PASS", "WARNING", "FS_UNRESOLVED"
    fs_val: float,
    preprocess_success: bool,
    morphology_guard_status: str,  # "PASSED", "DEGRADED_FALLBACK"
    feature_category: str,         # A, B, C, D
    readiness_score: float,
    quality_score: float,
    quality_band: str
) -> VerificationResult:
    """
    Evaluates whether all critical pipeline stages passed successfully.
    Issues the final Verification Verdict.
    """
    failure_reasons = []
    
    # Check 1: File Loader
    if not file_loader_success:
        failure_reasons.append("File Loader: Failed to decode, parse, or load signal data.")
        
    # Check 2: FS Detector (Critical)
    if fs_status == "FS_UNRESOLVED":
        failure_reasons.append("FS Detector: Sampling frequency is unresolved (cannot compute frequency features).")
        
    # Check 3: Preprocessor
    if not preprocess_success:
        failure_reasons.append("Preprocessor: Preprocessing failed (NaNs or empty array returned).")
        
    # Check 4: Morphology Guard
    if morphology_guard_status == "DEGRADED_FALLBACK":
        failure_reasons.append("Morphology Guard: Filtering severely distorted physiological waves, fell back to raw.")
        
    # Check 5: Feature Validator (Critical)
    if feature_category == "D" or readiness_score < 50.0:
        failure_reasons.append(f"Feature Validator: Signal features failed physiological plausibility (readiness: {readiness_score:.1f}%).")
        
    # Final Verdict
    if len(failure_reasons) > 0:
        overall_verdict = "Verification FAIL"
        wording = "Not ready for biomedical research"
    else:
        overall_verdict = "Verification PASS"
        wording = "Ready for biomedical research"
        
    return VerificationResult(
        file_name=file_name,
        overall_verdict=overall_verdict,
        readiness_pct=readiness_score,
        readiness_category=feature_category,
        quality_score=quality_score,
        quality_band=quality_band,
        fs_hz=fs_val,
        wording=wording,
        failure_reasons=failure_reasons
    )
