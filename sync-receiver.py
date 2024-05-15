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

sync = np.genfromtxt('sync-chirp.csv', delimiter=',')
#print(type(sync))
#print(sync)

#sync = np.pad(sync, (0,len(recording)-len(sync)))
correlation = scipy.signal.correlate(recording, sync)

plt.plot(np.absolute(correlation))
plt.show()


fftr = np.fft.fft(recording)
#print(recording)
#print(fftr)
#plt.plot(recording)
#plt.show()
#plt.plot(np.absolute(fftr))
#visualize.plot_fft(fftr, fs)