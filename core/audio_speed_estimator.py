import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import os
import sys

# Windows audible siren support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class AudioSpeedEstimator:
    """
    Acoustic Doppler Vehicle Speed Estimator.
    Extracts pass-by acoustics, detects Closest Point of Approach (CPA),
    and computes vehicle velocity via Doppler frequency shift equations.
    """
    def __init__(self, speed_of_sound_ms=343.0, road_setback_m=3.0):
        self.c = float(speed_of_sound_ms)       # Speed of sound in air (~343 m/s)
        self.d0 = float(road_setback_m)         # Distance from microphone to vehicle lane
        self.min_freq_hz = 120.0
        self.max_freq_hz = 2400.0

    def estimate_speed(self, wav_path):
        """
        Estimates vehicle speed from an audio recording.
        
        Args:
            wav_path: path to 16-bit PCM WAV file
            
        Returns:
            dict containing:
                - 'speed_kmh': float
                - 'cpa_timestamp': float (seconds)
                - 'f_approach': float (Hz)
                - 'f_recede': float (Hz)
                - 'snr_db': float
                - 'confidence': float (0.0 to 1.0)
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        fs, data = wavfile.read(wav_path)
        
        # If stereo, convert to mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Normalize to float32 [-1.0, 1.0]
        data = data.astype(np.float32)
        max_val = np.max(np.abs(data)) + 1e-7
        data = data / max_val
        duration = len(data) / float(fs)

        # 1. Bandpass filter to isolate vehicle acoustic band (tire & engine)
        b, a = signal.butter(4, [self.min_freq_hz / (fs / 2), self.max_freq_hz / (fs / 2)], btype='band')
        filtered = signal.filtfilt(b, a, data)

        # 2. Compute STFT Spectrogram
        nperseg = int(fs * 0.05)  # 50 ms window
        noverlap = int(nperseg * 0.75)
        f_bins, t_bins, Zxx = signal.stft(filtered, fs=fs, nperseg=nperseg, noverlap=noverlap)
        spec = np.abs(Zxx) ** 2

        # 3. Energy Envelope & Closest Point of Approach (CPA)
        # CPA occurs when sound intensity reaches maximum
        energy_env = np.sum(spec, axis=0)
        cpa_bin = np.argmax(energy_env)
        cpa_time = float(t_bins[cpa_bin])

        # 4. Spectral Centroid Trajectory
        freq_weights = f_bins[:, np.newaxis]
        centroid = np.sum(freq_weights * spec, axis=0) / (energy_env + 1e-7)

        # 5. Extract Approach vs Receding Frequencies
        # Look 1.2 seconds before CPA and 1.2 seconds after CPA
        t_pre = max(0.2, cpa_time - 1.2)
        t_post = min(duration - 0.2, cpa_time + 1.2)

        idx_pre = np.argmin(np.abs(t_bins - t_pre))
        idx_post = np.argmin(np.abs(t_bins - t_post))

        # Smooth centroid around approach and recede windows
        win_size = max(3, len(t_bins) // 25)
        smoothed = np.convolve(centroid, np.ones(win_size)/win_size, mode='same')

        f_approach = float(np.median(smoothed[max(0, idx_pre - 5):idx_pre + 5]))
        f_recede = float(np.median(smoothed[max(0, idx_post - 5):idx_post + 5]))

        # Ensure f_approach > f_recede due to Doppler shift
        if f_approach <= f_recede:
            # Fallback to harmonic peak detection
            peak_freqs = f_bins[np.argmax(spec, axis=0)]
            f_approach = float(np.median(peak_freqs[max(0, idx_pre - 5):idx_pre + 5]))
            f_recede = float(np.median(peak_freqs[max(0, idx_post - 5):idx_post + 5]))

        # 6. Physical Velocity from Doppler Ratio:
        # v = c * (f_app - f_rec) / (f_app + f_rec)
        delta_f = max(1.0, f_approach - f_recede)
        sum_f = max(10.0, f_approach + f_recede)
        ratio = delta_f / sum_f
        
        # Velocity in m/s, clamped to realistic vehicle physics
        v_ms = self.c * ratio
        speed_kmh = float(v_ms * 3.6)
        speed_kmh = np.clip(speed_kmh, 25.0, 130.0)

        # Confidence metric based on signal-to-noise ratio
        noise_floor = np.median(energy_env)
        peak_energy = energy_env[cpa_bin]
        snr_db = float(10 * np.log10((peak_energy + 1e-7) / (noise_floor + 1e-7)))
        confidence = float(np.clip(snr_db / 25.0, 0.3, 0.98))

        return {
            "speed_kmh": round(speed_kmh, 1),
            "cpa_timestamp": round(cpa_time, 2),
            "f_approach": round(f_approach, 1),
            "f_recede": round(f_recede, 1),
            "snr_db": round(snr_db, 1),
            "confidence": round(confidence, 2)
        }

    @staticmethod
    def play_speed_violation_alarm(frequency_hz=1400, duration_ms=250):
        """
        Emits an audible emergency warning tone when an over-speeding
        or wrong-way infraction is triggered.
        """
        if HAS_WINSOUND:
            try:
                winsound.Beep(int(frequency_hz), int(duration_ms))
            except Exception:
                pass
