import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

FS = 100
F_HIGH_PASS = 0.5
F_LOW_PASS = 40.0
F_NOTCH = 50.0
QUALITY_FACTOR = 30


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, data)


def notch_filter(data, notch_freq, fs, Q):
    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, data)


def clean_ecg_signal(signal, fs=FS, f_high_pass=F_HIGH_PASS, 
                    f_low_pass=F_LOW_PASS, f_notch=F_NOTCH):
    num_samples, num_leads = signal.shape
    signal_filtered = np.zeros_like(signal, dtype=np.float32)
    
    for j in range(num_leads):
        lead_signal = signal[:, j]
        
        lead_filtered = butter_bandpass_filter(
            lead_signal, f_high_pass, f_low_pass, fs
        )
        
        if f_notch > 1.0:
            lead_filtered = notch_filter(
                lead_filtered, f_notch, fs, QUALITY_FACTOR
            )
        
        signal_filtered[:, j] = lead_filtered
    
    return signal_filtered


def normalize_signal(signal, mean=None, std=None):
    if mean is None:
        mean = np.mean(signal, axis=0, keepdims=True)
    if std is None:
        std = np.std(signal, axis=0, keepdims=True)
    
    normalized = (signal - mean) / (std + 1e-8)
    
    return normalized, mean, std


def preprocess_ecg(signal, metadata, global_mean=None, global_std=None,
                  apply_filtering=True):
    if apply_filtering:
        signal_clean = clean_ecg_signal(signal)
    else:
        signal_clean = signal
    
    if global_mean is not None and global_std is not None:
        signal_norm, _, _ = normalize_signal(signal_clean, global_mean, global_std)
    else:
        signal_norm, _, _ = normalize_signal(signal_clean)
    
    metadata_array = np.array(metadata, dtype=np.float32)
    
    return signal_norm, metadata_array
