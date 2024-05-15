import sounddevice as sd
import visualize
import numpy as np
import scipy.signal

from ctypes.util import find_library
find_library('portaudio')
                                                   

import matplotlib.pyplot as plt
seconds = 4
fs = 44100

input = input('press space')
recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()
recording = recording.flatten()
#print(type(recording))
#print(np.shape(recording))

sync = np.genfromtxt('sync.csv', delimiter=',')
#print(type(sync))
#print(sync)

#sync = np.pad(sync, (0,len(recording)-len(sync)))
correlation = scipy.signal.correlate(recording, sync)

peak_correlation = np.max(correlation)
position = np.where(correlation ==peak_correlation) - len(sync)
print("position:", position)
print(len(sync))
chirp = recording[position:position+len(sync)]
fftr = np.fft.fft(chirp)

plt.plot(correlation)
plt.show()

visualize.plot_fft(fftr, fs)


#print(recording)
#print(fftr)
#plt.plot(recording)
#plt.show()
#plt.plot(np.absolute(fftr))
#visualize.plot_fft(fftr, fs)