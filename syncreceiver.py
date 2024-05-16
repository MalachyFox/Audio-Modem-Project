import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
from ctypes.util import find_library
find_library('portaudio')
                                                   

import matplotlib.pyplot as plt
seconds = 6
fs = 44100

f0 = 1000
f1 = 2000

sync = playsound.gen_chirp(f0,f1,fs,1)
sync = playsound.double_signal(sync)
#sync = playsound.gen_chirp(500,1500,fs,1)
#playsound.save_signal(signal,'sync-chirp-low-long')

input = input('press space')
recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()
recording = recording.flatten()
#print(type(recording))
#print(np.shape(recording))

#sync = np.genfromtxt('sync-chirp-low.csv', delimiter=',')
#print(type(sync))
#print(sync)

block_length = 1000

#sync = np.pad(sync, (0,len(recording)-len(sync)))
correlation = scipy.signal.correlate(recording, sync)

peak_correlation = np.max(correlation)
position = int(np.where(correlation ==peak_correlation)[0]) - len(sync)
print("position:", position)
print(len(sync))
plt.plot(recording)
plt.show()
chirp = recording[position + len(sync)//2 :position+len(sync)]
fftr = np.fft.fft(chirp)
sync1 = sync[len(sync)//2:]
ffts = np.fft.fft(sync1)

#data = recording[position+block_length:position+block_length*2]

plt.plot(correlation)
plt.show()
visualize.plot_fft(ffts, fs)
visualize.plot_fft(fftr, fs)

print(len(fftr),len(ffts))

channel = fftr / ffts#fftr[f0:f1] / ffts[f0:f1]
visualize.plot_fft(channel, fs)
visualize.plot_constellation(channel)

impulse = np.fft.ifft(fftr/ffts)
plt.plot(impulse)
plt.show()






#print(recording)
#print(fftr)
#plt.plot(recording)
#plt.show()
#plt.plot(np.absolute(fftr))
#visualize.plot_fft(fftr, fs)