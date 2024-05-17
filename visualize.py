from math import inf
from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_channel(channel):
    plt.stem(range(len(channel)),channel)
    plt.show()

def plot_fft(fft,fs,f0,f1):
    n_samples = len(fft)
    dur = n_samples/fs
    x = np.linspace(f0,f1,(n_samples//fs) * (f1-f0))
    fft = fft[int(f0*dur):int(f1*dur)]
    fig, ax = plt.subplots(2)
    ax[0].title.set_text('Frequency Domain')
    ax[0].set_xlabel('f / Hz')
    ax[0].set_ylabel('Log10 Ambplitude')
    #ax[0].plot(x,np.log10(np.absolute(fft)))
    ax[0].plot(x,np.log(np.absolute(fft)))
    #ax[1].plot(x,np.absolute(fft))
    #ax[2].plot(x,np.angle(fft))
    ax[1].scatter(x,np.angle(fft),s=4)
    ax[1].set_xlabel("f / Hz")
    ax[1].set_ylabel('Phase / rad')
    plt.savefig(f"FFT-{f0}-{f1}-{datetime.datetime.now()}")
    plt.show()
    

def plot_constellation(fft):
    l=abs(np.max(fft))*1.2
    r = np.real(fft)
    i = np.imag(fft)
    plt.scatter(r,i,s=4)
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.axis('scaled')
    if l == np.inf or l==np.NaN:
        l=100
    l = 300
    plt.ylim(-l, l)
    plt.xlim(-l, l)
    plt.savefig(f"CONSTELLATION-{datetime.datetime.now()}")
    plt.show()