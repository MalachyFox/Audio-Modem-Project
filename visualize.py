from math import inf
from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_channel(channel):
    plt.stem(range(len(channel)),channel)
    plt.show()

def plot_fft(fft,fs,title=""):
    n_samples = len(fft)
    x = list(range(n_samples))
    fig, ax = plt.subplots(2)
    ax[0].title.set_text('Frequency Domain')
    ax[0].set_xlabel('f / Hz')
    ax[0].set_ylabel('Log10 Ambplitude')
    ax[0].plot(x,np.log(np.absolute(fft)))
    ax[1].scatter(x,np.angle(fft),s=4)
    ax[1].set_xlabel("f / Hz")
    ax[1].set_ylabel('Phase / rad')
    if title != "":
        plt.savefig(f"test_figures/{title}-fft.png")
    plt.show()
    

def plot_constellation(fft,colours,title=""):
    l=abs(np.max(fft))*1.2
    r = np.real(fft)
    i = np.imag(fft)
    plt.scatter(r,i,s=6,c=colours,alpha=0.3)
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.axis('scaled')
    avg = np.average(np.absolute(fft))
    l = avg * 2
    plt.ylim(-l, l)
    plt.xlim(-l, l)
    if title != "":
        plt.savefig(f"test_figures/{title}-con.png")
    plt.show()

def big_plot(blocks,fs,colours,title=""):

    fig, axs = plt.subplots(2,len(blocks),sharex='row',sharey='row')
    fig.set_size_inches(18, 6)

    for i in range(len(blocks)):

        fft = blocks[i]
        col = colours[i*len(fft):(i+1)*len(fft)]
        

        x = list(range(len(fft)))

        axs[1,i].scatter(x,np.angle(fft),s=2,c=col,alpha=0.1)
        axs[1,i].set_xlabel("f / Hz")
        axs[1,i].set_ylabel('Phase / rad')


        r = np.real(fft)
        im = np.imag(fft)

        axs[0,i].scatter(r,im,s=2,c=col,alpha=0.1)
        axs[0,i].axhline(0, color='gray')
        axs[0,i].axvline(0, color='gray')
        axs[0,i].axis('scaled')

        avg = np.average(np.absolute(fft))
        l = avg * 2
        axs[0,i].set_ylim(-l, l)
        axs[0,i].set_xlim(-l, l)

    if title != "":
        plt.savefig(f"test_figures/{title}-big.png",dpi=300)

    plt.show()