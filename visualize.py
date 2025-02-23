from math import inf
from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import rc

def plot_channel(channel):
    #plt.stem(range(len(channel)),channel)
    plt.plot(channel)
    plt.show()

def plot_fft(fft,fs,colours='b',title=""):
    freqs = np.fft.fftfreq(len(fft),1/fs)
    fig, ax = plt.subplots(2)
    ax[0].title.set_text('Frequency Domain')
    ax[0].set_xlabel('f / Hz')
    ax[0].set_ylabel('Log10 Amplitude')
    ax[0].plot(freqs,np.log10(np.absolute(fft)))
    ax[1].scatter(freqs,np.angle(fft),s=4)
    ax[1].set_xlabel("f / Hz")
    ax[1].set_ylabel('Phase / rad')
    if title != "":
        plt.savefig(f"test_figures/{title}-fft.png")
    plt.show()
    

def plot_constellation(fft,colours=None,title=""):
    l=np.max(np.absolute(fft))*1.2
    r = np.real(fft)
    i = np.imag(fft)
    if colours !=None:
        plt.scatter(r,i,s=1,c=colours,alpha=1)
    else:
        plt.scatter(r,i,s=1,alpha=1)
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.axis('scaled')
    avg = np.average(np.absolute(fft))
    l = avg * 3
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
        axs[1,i].set_ylim(-np.pi, np.pi)
        axs[1,i].set_yticks(np.linspace(-np.pi,np.pi,9))

        x = list(range(len(fft)))

        axs[1,i].scatter(x,np.angle(fft),s=8,alpha=1,c=col)
        axs[1,i].set_xlabel(r"Information bin index",fontname="serif")
        axs[1,i].set_ylabel(r'Phase (rad)',fontname="serif")


        r = np.real(fft)
        im = np.imag(fft)

        axs[0,i].scatter(r,im,s=8,alpha=1,c=col)
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