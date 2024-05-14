import sounddevice as sd
import visualize
import numpy as np

from ctypes.util import find_library
find_library('portaudio')
                                                   

import matplotlib.pyplot as plt
seconds = 3
fs = 44100

input = input('press space')
recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()
recording = recording.flatten()
print(type(recording))

fftr = np.fft.fft(recording)
print(recording)
print(fftr)
plt.plot(recording)
plt.show()
#plt.plot(np.absolute(fftr))
visualize.plot_fft(fftr)