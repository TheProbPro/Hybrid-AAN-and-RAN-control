import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfilt_zi, tf2sos
import time

class rt_filtering:
    def __init__(self, sample_rate, lp_cutoff=450, hp_cutoff=20, order=4, mains=50.0, notch_bw=1.0):
        self.fs = sample_rate       # Sample rate in Hz
        self.nyq = 0.5 * self.fs    # Nyquist frequency

        # --- design filters (SOS) ---
        # This is for rectification after Bandpass
        self.lp_sos = butter(2, 3 / self.nyq, btype='lowpass', output='sos')
        self.lp_zi  = sosfilt_zi(self.lp_sos) * 0.0 
        
        self.hp_sos = butter(order, hp_cutoff / self.nyq, btype='highpass', output='sos')
        self.hp_zi  = sosfilt_zi(self.hp_sos) * 0.0

        self.bandpass_sos = butter(order, [hp_cutoff / self.nyq, lp_cutoff / self.nyq], btype='bandpass', output='sos')
        self.bandpass_zi  = sosfilt_zi(self.bandpass_sos) * 0.0

        self.bandstop_sos = butter(order, [ (mains-2)/self.nyq, (mains+2)/self.nyq ], btype='bandstop', output='sos')
        self.bandstop_zi  = sosfilt_zi(self.bandstop_sos) * 0.0

        # Notch (use fs= to specify Hz directly)
        Q = mains / notch_bw
        b_notch, a_notch = iirnotch(mains, Q, fs=self.fs)
        self.notch_sos = tf2sos(b_notch, a_notch)
        self.notch_zi  = sosfilt_zi(self.notch_sos)

    def Windowed_RMS(self, filtered_data, window_size=50):
        rms_values = []
        for i in range(0, len(filtered_data), window_size):
            window = filtered_data[i:i+window_size]
            rms = np.sqrt(np.mean(np.square(np.abs(window))))
            rms_values.append(rms)
        return rms_values
    
    def RMS(self, window, window_size=50):
        if len(window) < window_size:
            return window[-1]
        return float(np.sqrt(np.mean(np.square(np.abs(window)))))
    
    def bandpass(self, data):
        y, self.bandpass_zi = sosfilt(self.bandpass_sos, data, zi=self.bandpass_zi)
        return y
    
    def notch(self, data):
        y, self.notch_zi = sosfilt(self.notch_sos, data, zi=self.notch_zi)
        return y
    
    def lowpass(self, data):
        y, self.lp_zi = sosfilt(self.lp_sos, data, zi=self.lp_zi)
        return y

class rt_desired_Angle_lowpass:
    def __init__(self, sample_rate, lp_cutoff=3, order=2):
        self.fs = sample_rate       # Sample rate in Hz
        self.nyq = 0.5 * self.fs    # Nyquist frequency

        # --- design filters (SOS) ---
        self.lp_sos = butter(order, lp_cutoff / self.nyq, btype='lowpass', output='sos')
        self.lp_zi  = sosfilt_zi(self.lp_sos) * 0.0

    def lowpass(self, data):
        y, self.lp_zi = sosfilt(self.lp_sos, data, zi=self.lp_zi)
        return y

    def reset(self):
        self.lp_zi  = sosfilt_zi(self.lp_sos) * 0.0