import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, periodogram
from scipy.stats import skew, kurtosis
from typing import Tuple, Dict, Any, List

def calculate_derivatives(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 1st derivative (VPG) and 2nd derivative (APG) of the PPG signal.
    """
    dt = 1.0 / fs
    vpg = np.gradient(signal, dt)
    apg = np.gradient(vpg, dt)
    return vpg, apg

def calculate_entropy(signal: np.ndarray, bins: int = 50) -> float:
    """
    Computes Shannon entropy of the signal amplitude distribution.
    """
    # Normalize signal to avoid bin issues
    sig_min, sig_max = np.min(signal), np.max(signal)
    if sig_max - sig_min < 1e-9:
        return 0.0
    hist, _ = np.histogram(signal, bins=bins, range=(sig_min, sig_max), density=True)
    # Convert density to probability
    probs = hist / np.sum(hist)
    # Filter out zeros
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def calculate_snr_and_drift(signal: np.ndarray, fs: float) -> Tuple[float, float]:
    """
    Estimates SNR (ratio of power in 0.5-4.0 Hz band to noise power outside)
    and baseline drift (std dev of low-frequency components < 0.5 Hz).
    """
    n = len(signal)
    if n < 2:
        return 0.0, 0.0
        
    # Subtract mean first
    sig_detrend = signal - np.mean(signal)
    
    # Calculate periodogram
    freqs, psd = periodogram(sig_detrend, fs)
    
    # Signal band: 0.5 to 4.0 Hz
    sig_mask = (freqs >= 0.5) & (freqs <= 4.0)
    # Noise band: > 0.5 Hz but outside 0.5-4.0 Hz (i.e. > 4.0 Hz)
    # High frequency noise is above 4 Hz. Very low frequency (<0.5 Hz) is baseline wander.
    noise_mask = (freqs > 4.0)
    
    p_signal = np.sum(psd[sig_mask])
    p_noise = np.sum(psd[noise_mask])
    
    if p_noise <= 0:
        snr = 30.0 # High default if noise is virtually 0
    else:
        snr = 10.0 * np.log10(p_signal / p_noise)
        
    # Baseline drift: apply low pass butterworth filter at 0.5 Hz
    # If the signal is too short to filter, use standard deviation of a moving average
    drift_std = 0.0
    try:
        if n > 3 * int(fs / 0.5):
            nyq = 0.5 * fs
            cutoff = 0.5 / nyq
            b, a = butter(2, cutoff, btype='low')
            baseline = filtfilt(b, a, signal)
            drift_std = float(np.std(baseline))
        else:
            # Short signal approximation
            window = max(3, int(fs))
            baseline = pd.Series(signal).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').to_numpy()
            drift_std = float(np.std(baseline))
    except Exception:
        drift_std = float(np.std(signal - sig_detrend)) # Fallback
        
    return float(snr), float(drift_std)

def estimate_fs_from_data(signal: np.ndarray) -> Dict[str, Any]:
    """
    Estimates sampling rate Fs by finding the heartbeat period in samples using:
    1. FFT-based peak spacing
    2. Autocorrelation (ACF) first peak lag
    3. Power Spectral Density (PSD) peak
    Then matches against candidates [30, 60, 100, 125, 250, 500, 1000] Hz.
    Returns the estimated Fs, the confidence level, and individual estimates.
    """
    # Detrend and crop if signal is too long (to prevent infinite correlation time)
    max_len = 5000
    if len(signal) > max_len:
        mid = len(signal) // 2
        signal_slice = signal[mid - max_len//2 : mid + max_len//2]
    else:
        signal_slice = signal
        
    n_slice = len(signal_slice)
    candidates = [30, 60, 100, 125, 250, 500, 1000]
    
    # Detrend
    sig_detrend = signal_slice - np.mean(signal_slice)
    
    # 1. ACF estimate
    acf = np.correlate(sig_detrend, sig_detrend, mode='full')[n_slice-1:]
    if acf[0] > 0:
        acf = acf / acf[0]
        
    # Find peaks in ACF starting from lag 10 to avoid zero lag peak
    # Max lag is min(n_slice/2, 2000)
    max_lag = min(n_slice // 2, 2000)
    acf_peaks, _ = find_peaks(acf[:max_lag], distance=8, prominence=0.05)
    
    acf_period = None
    if len(acf_peaks) > 0:
        acf_period = acf_peaks[0] # First peak lag
        
    # 2. FFT estimate
    fft_vals = np.abs(np.fft.rfft(sig_detrend))
    # Exclude DC and very low frequencies
    fft_peaks, _ = find_peaks(fft_vals, distance=2)
    fft_period = None
    if len(fft_peaks) > 0:
        # Find highest peak in FFT
        highest_fft_peak = fft_peaks[np.argmax(fft_vals[fft_peaks])]
        if highest_fft_peak > 0:
            fft_period = n_slice / highest_fft_peak
            
    # 3. PSD estimate (periodogram)
    # Using raw indices since we don't know Fs, so we assume Fs = 1 (index spacing)
    # PSD peak index will correspond to period in samples = n_slice / k_psd
    psd_vals = fft_vals ** 2
    psd_peaks, _ = find_peaks(psd_vals, distance=2)
    psd_period = None
    if len(psd_peaks) > 0:
        highest_psd_peak = psd_peaks[np.argmax(psd_vals[psd_peaks])]
        if highest_psd_peak > 0:
            psd_period = n_slice / highest_psd_peak
            
    # Voting system:
    # For each candidate Fs, calculate heart rate = (Fs / period_samples) * 60
    # A physiological rest HR is around 50 - 120 bpm (target = 72 bpm).
    # Score candidate Fs = exp(-0.5 * ((hr - 72) / 25)^2). If hr out of [40, 180], score = 0.
    
    def score_fs(fs_val: float, period_samples: float) -> float:
        if period_samples <= 0:
            return 0.0
        hr = (fs_val / period_samples) * 60.0
        if hr < 40.0 or hr > 180.0:
            return 0.0
        return float(np.exp(-0.5 * ((hr - 72.0) / 25.0) ** 2))
        
    votes = {}
    methods = {"acf": acf_period, "fft": fft_period, "psd": psd_period}
    method_decisions = {}
    
    for name, p in methods.items():
        if p is None or p <= 0:
            method_decisions[name] = None
            continue
            
        best_cand = None
        best_score = -1.0
        for cand in candidates:
            sc = score_fs(cand, p)
            if sc > best_score:
                best_score = sc
                best_cand = cand
                
        method_decisions[name] = best_cand
        
    # Count votes
    valid_decisions = [v for v in method_decisions.values() if v is not None]
    if not valid_decisions:
        # If no method could resolve, fallback to standard 100 Hz
        return {
            "estimated_fs": 100.0,
            "confidence": 0.0,
            "votes": method_decisions,
            "agreement_score": 0.0
        }
        
    # Select majority vote
    from collections import Counter
    counts = Counter(valid_decisions)
    estimated_fs, vote_count = counts.most_common(1)[0]
    
    # Agreement score is ratio of agreeing estimators to total active estimators
    agreement_score = vote_count / len(valid_decisions)
    
    return {
        "estimated_fs": float(estimated_fs),
        "votes": method_decisions,
        "agreement_score": float(agreement_score)
    }
