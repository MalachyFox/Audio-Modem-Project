import sounddevice as sd

from ctypes.util import find_library
find_library('portaudio')
                                                   

import matplotlib.pyplot as plt
seconds = 5
fs = 44100


recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()

print(recording)

plt.plot(recording)
plt.show()