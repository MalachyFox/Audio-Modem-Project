import sounddevice as sd
import numpy as np
from scipy.signal import chirp
import visualize
import matplotlib.pyplot as plt
import csv
import soundfile as sf

def gen_sine(f,fs,duration):

    t = np.arange(duration * fs)
    transmitted_signal = np.sin(2 * np.pi * t * f / fs)
    return transmitted_signal

def gen_chirp(N0,N1,fs,num_samples,block_length):
    t = np.linspace(0, num_samples/fs, num_samples,endpoint=False)  # Time array
    signal = chirp(t, (N0*fs)/block_length, num_samples/fs, (N1*fs)/block_length, method='linear')
    return signal

def gen_random(samples):
    signal = np.random.random(samples) * 2 - 1
    return signal

def save_signal(signal,fs,filename):
    if filename[-4:] == ".wav":
        sf.write(filename,signal,fs)
    else:
        with open(filename, "w") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(signal)
    return

def play_signal(signal,fs):
    input("press enter to play")
    sd.play(signal,fs)
    sd.wait()

def load_signal(filename):
    if filename[-4:] == ".wav":
        signal, fs = sf.read(filename)
    else:
        signal = np.genfromtxt(filename,delimiter=',')
    return signal

def super_sine(f_array, fs,duration=1):
    print(f_array)
    t = np.arange(duration * fs)
    transmitted_signal = np.zeros(len(t))
    for f in f_array:
        transmitted_signal += np.sin(2 * np.pi * t * f / fs)
    return transmitted_signal/len(f_array)

def double_signal(signal):
    return np.concatenate((signal,signal))


if __name__ == "__main__":
    pass
    
#signal = super_sine(np.linspace(500,1000,50),fs,2)
#print(signal[100:200])
#save_signal(signal,'sync-chirp-low.csv')
#play_signal(signal,fs)




# #signal = play_tone(2000,fs,10)
# #signal = play_chirp(20,20000,fs,10)

# # Generate linear chirp signal

# # signal = save_sync(fs)

#fft = np.fft.fft(signal)

#visualize.plot_fft(fft,fs)






