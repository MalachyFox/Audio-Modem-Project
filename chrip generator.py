import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
import soundfile as sf

# Parameters
duration =   # Duration of the chirp signal in seconds
f0 = 100  # Initial frequency of the chirp in Hz
f1 = 1000  # Final frequency of the chirp in Hz
fs = 44100  # Sampling frequency
t = np.linspace(0, duration, int(fs * duration))  # Time array

# Generate linear chirp signal
signal = chirp(t, f0, duration, f1, method='linear')

plt.figure()
plt.plot(t, signal)
plt.title('Linear Chirp Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Scale to 16-bit PCM range (-32768 to 32767)
signal = np.int16(signal * 32767)

# Save the signal as a WAV file
sf.write('/Users/lit./Desktop/gf3/chirp_signal.wav', signal, fs)