import scipy
import numpy as np
import matplotlib.pyplot as plt

fs = 48000
block_length = 4096
t1 = block_length/fs
f0 = 1000
f1 = 10000

N0 = 85 # int((f0/fs)*block_length)
N1 = 850 # int((f1/fs)*block_length)

t = np.linspace(0,block_length/fs,block_length,endpoint=False)
signal = scipy.signal.chirp(t,f0,block_length/fs,f1)

sig_fft = np.fft.rfft(signal,block_length)
sig_fft_freq = np.fft.rfftfreq(block_length,1/fs)
plt.plot(sig_fft_freq,np.log10(np.absolute(sig_fft)))
plt.show()
plt.plot(np.log10(np.absolute(sig_fft[85:850])))
plt.show()