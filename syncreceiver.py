import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
from ctypes.util import find_library
find_library('portaudio')
                                                   

import matplotlib.pyplot as plt
seconds = 7
fs = 44100

f0 = 5500
block_length = 2000
f1 = f0 + block_length

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



#sync = np.pad(sync, (0,len(recording)-len(sync)))
correlation = np.absolute(scipy.signal.correlate(recording, sync))

peak_correlation = np.max(correlation)
position = int(np.where(correlation ==peak_correlation)[0]) - len(sync)
print("position:", position)
print(len(sync))
plt.plot(recording)
plt.show()
chirp = recording[position + len(sync)//2 :position+len(sync)]
#plt.plot(chirp)
#plt.show()
fftr = np.fft.fft(chirp)
sync1 = sync[len(sync)//2:]
#plt.plot(sync1)
#plt.show()
ffts = np.fft.fft(sync1)

data = recording[position+len(sync)+fs:position+len(sync)+fs*2]
#data = np.pad(data,(0,fs-len(data)))
#plt.plot(correlation)
#plt.show()

vf0 = f0-300
vf1 = f1+300
#visualize.plot_fft(ffts, fs,vf0,vf1)
#visualize.plot_fft(fftr, fs,vf0,vf1)

print(len(fftr),len(ffts))

channel = fftr / ffts#fftr[f0:f1] / ffts[f0:f1]
channel = channel[f0:f1]
channel = np.pad(channel,(f0,fs-f1))
visualize.plot_fft(channel, fs,vf0,vf1)
#visualize.plot_constellation(channel)

impulse = np.fft.irfft(channel)
#plt.plot(impulse)
#plt.show()

data_fft = np.fft.fft(data)
data_fft = data_fft[f0:f1]
print(len(data_fft))
visualize.plot_fft(data_fft,fs,0,fs)
visualize.plot_constellation(data_fft)
data_fft = data_fft/(channel[f0:f1])
visualize.plot_fft(data_fft,fs,0,fs)
visualize.plot_constellation(data_fft)






#print(recording)
#print(fftr)
#plt.plot(recording)
#plt.show()
#plt.plot(np.absolute(fftr))
#visualize.plot_fft(fftr, fs)