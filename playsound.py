import sounddevice as sd
import numpy as np
from scipy.signal import chirp
import visualize
import matplotlib.pyplot as plt
import csv

fs = 44100

def gen_sine(f,fs,duration):

    t = np.arange(duration * fs)
    transmitted_signal = np.sin(2 * np.pi * t * f / fs)
    return transmitted_signal

def gen_chirp(f0,f1,fs,duration):
    t = np.linspace(0, duration, int(fs * duration))  # Time array
    transmitted_signal = chirp(t, f0, duration, f1, method='linear')
    return transmitted_signal

def gen_random(samples):
    signal = np.random.random(samples) * 2 - 1

    return signal

def save_signal(signal,filename):
    with open(filename, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(signal)
    return

def play_signal(signal,fs):
    input("press enter to play")
    sd.play(signal,fs)
    sd.wait()

def load_signal(filename):
    signal = np.genfromtxt(filename,delimiter=',')
    return signal

def super_sine(f_array, fs,duration):
    print(f_array)
    t = np.arange(duration * fs)
    transmitted_signal = np.zeros(len(t))
    for f in f_array:
        transmitted_signal += np.sin(2 * np.pi * t * f / fs)
    return transmitted_signal/len(f_array)


#signal = gen_chirp( 500,1500,fs,1)
signal = super_sine(np.linspace(500,1000,50),fs,2)
print(signal[100:200])
#save_signal(signal,'sync-chirp-low.csv')
play_signal(signal,fs)




#signal = play_tone(2000,fs,10)
#signal = play_chirp(20,20000,fs,10)

# Generate linear chirp signal

# signal = save_sync(fs)

#fft = np.fft.fft(signal)

#visualize.plot_fft(fft,fs)






