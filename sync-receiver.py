import sounddevice as sd                    

import matplotlib.pyplot as plt
seconds = 5
fs = 44100


recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()

print(recording)

plt.plot(recording)
plt.show()