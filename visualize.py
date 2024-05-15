from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt

def plot_channel(channel):
    plt.stem(range(len(channel)),channel)
    plt.show()

def plot_fft(fft,fs):
    n_samples = len(fft)
    dur = n_samples/fs
    x = np.linspace(0,fs,n_samples)
    fig, ax = plt.subplots(2)
    ax[0].plot(x,np.absolute(fft))
    ax[1].plot(x,np.angle(fft))
    plt.show()

def plot_constellation(fft):
    l=abs(np.max(fft))*1.2
    r = np.real(fft)
    i = np.imag(fft)
    plt.scatter(r,i)
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.axis('scaled')
    plt.ylim(-l, l)
    plt.xlim(-l, l)
    plt.show()