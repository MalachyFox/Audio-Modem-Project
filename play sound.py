import sounddevice as sd
import numpy as np
from scipy.signal import chirp
import visualize
import matplotlib.pyplot as plt
import csv

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

def generate_sync(fs):
    signal = np.random.random(10000) * 2 - 1
    sd.play(signal, samplerate=fs)
    sd.wait()
    with open("sync.csv", "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(signal)
    return signal

input = input("press enter")
signal = np.genfromtxt('sync.csv',delimiter=',')
sd.play(signal,fs)
sd.wait()


#signal = play_tone(2000,fs,10)
#signal = play_chirp(20,20000,fs,10)

# Generate linear chirp signal

# signal = generate_sync(fs)

fft = np.fft.fft(signal)

visualize.plot_fft(fft,fs)






