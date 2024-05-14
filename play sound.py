import sounddevice as sd
import numpy as np
from scipy.signal import chirp

# Parameters
duration = 1.0  # Duration of the chirp signal in seconds
f0 = 100  # Initial frequency of the chirp in Hz
f1 = 1000  # Final frequency of the chirp in Hz
fs = 44100  # Sampling frequency
t = np.linspace(0, duration, int(fs * duration))  # Time array

# Generate linear chirp signal
transmitted_signal = chirp(t, f0, duration, f1, method='linear')

sd.play(transmitted_signal, samplerate=fs)
sd.wait()