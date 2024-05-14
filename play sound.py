import sounddevice as sd
import numpy as np
from scipy.signal import chirp
import visualize
import matplotlib.pyplot as plt

fs = 44100

def play_tone(f,fs,duration):

    t = np.arange(duration * fs)
    transmitted_signal = np.sin(2 * np.pi * t * f / fs)
    sd.play(transmitted_signal, samplerate=fs)
    sd.wait()
    return transmitted_signal

def play_chirp(f0,f1,fs,duration):
    t = np.linspace(0, duration, int(fs * duration))  # Time array
    transmitted_signal = chirp(t, f0, duration, f1, method='linear')
    sd.play(transmitted_signal, samplerate=fs)
    sd.wait()
    return transmitted_signal

input = input("press enter")
#signal = play_tone(2000,fs,10)
signal = play_chirp(2000,5000,fs,2)

# Generate linear chirp signal
 #

chirp_fft = np.fft.fft(signal)

visualize.plot_fft(chirp_fft,fs)

def generate_sync(f_start, f_stop, amplitude):
    return