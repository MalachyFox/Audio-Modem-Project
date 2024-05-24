from sys import prefix
from turtle import goto
from uu import encode
import bitarray
import numpy as np
from graycode import gray_code_to_tc
import visualize as v
import playsound as ps
import random
from matplotlib import pyplot as plt
from bitstring import BitArray

# with open('weekend-challenge/parsed.tiff',"rb") as file:
#      file_binary = file.read()

# binary = BitArray(file_binary).bin



### STANDARD ###
fs = 48000
block_length = 4096 
bits_per_value = 2
prefix_length = 512 
N0 = 85
N1 = 850
###
recording_time = 4
chirp_factor = 16
tracking_bins = 15
num_blocks = 1000
###
used_bins = N1 - N0
chirp_length = block_length * chirp_factor
used_bins_data = used_bins - tracking_bins
###
play = False
save = False


n=12
d_v = 3
d_c = 6


M = 2**bits_per_value


def random_binary(N):
    random.seed(1)
    return ''.join(random.choices(["0","1"], k=N))

def correct_binary_length(binary):
    one = len(binary)%bits_per_value
    binary = binary +  (bits_per_value - one)%bits_per_value * "0" # makes sure binary can be divided into values
    two = len(binary)//bits_per_value%used_bins
    binary = binary +  (used_bins - two)%(used_bins) * bits_per_value * "0" # makes sure binary can be divided into blocks
    return binary

def binary_str_to_symbols(binary):
    symbols = []
    for i in range(len(binary)//bits_per_value):
        symbols.append(binary[bits_per_value*i:bits_per_value*(i+1)])
    return symbols

def symbols_to_phases(symbols):
    phases = []
    for symbol in symbols:

        b_int = gray_code_to_tc(int(symbol,2))
        b_phase = (b_int + 0.5)*2*np.pi /M    #b_int + 0.5? ??
        if b_phase > np.pi:
            b_phase = -(2*np.pi - b_phase)
        phases.append(b_phase)
    return phases

def phases_to_blocks(phases):

    blocks = []
    for p in range(int(len(phases)/used_bins)):
        blocks.append(phases[used_bins * p:used_bins*(1+p)])
    return blocks

def blocks_to_blocks_fft(blocks):

    blocks_fft = []
    for block in blocks:
        block_f_d = np.zeros(block_length//2 + 1,dtype=np.complex_)  #f1
        for i in range(used_bins):
            block_f_d[N0 + i] = np.exp(1j*block[i])
        blocks_fft.append(block_f_d)
    
    return blocks_fft

def blocks_fft_to_signal(blocks_fft):

    transmission = []
    for block in blocks_fft:
        block_symbol = np.fft.irfft(block)
        block_symbol = np.concatenate((block_symbol[-prefix_length:],block_symbol))
        transmission = np.concatenate((transmission,block_symbol))

    transmission = transmission /np.max(transmission)
    chirp = ps.gen_chirp(N0,N1,fs,chirp_length,block_length)
    chirp = np.concatenate((chirp[-prefix_length:],chirp))
    chirp = chirp/np.max(chirp)
    transmission = np.concatenate((chirp,transmission))
    
    return transmission

def add_tracking(binary,tracking_bins=tracking_bins):
    output = ""
    bl = used_bins_data*bits_per_value
    spacing = (bl//(2*tracking_bins))*2
    for b in [binary[i*bl:(i+1)*bl] for i in range(num_blocks)]:
        b_new = ""
        for s in range(tracking_bins):
            
            if s == tracking_bins - 1:
                chunk = b[s*spacing:]
            else:
                chunk = b[s*spacing:(s+1)*spacing]
            chunk = "00" + chunk
            b_new += chunk
        output += b_new
    return output 


if __name__ == "__main__":
    
    binary = random_binary(used_bins_data*bits_per_value*num_blocks)
    len_binary_data = len(binary)
    binary = add_tracking(binary)
    binary = correct_binary_length(binary)
    len_binary = len(binary)
    symbols = binary_str_to_symbols(binary)
    phases = symbols_to_phases(symbols)
    blocks = phases_to_blocks(phases)
    blocks_fft = blocks_to_blocks_fft(blocks)
    signal = blocks_fft_to_signal(blocks_fft)

    print()
    print(f"fs:       ",fs)
    print(f'N0:       ',N0)
    print(f'N1:       ',N1)
    print(f"blck len: ",block_length)
    print(f"prfx len: ",prefix_length)
    print(f"num blcks:",len(blocks))
    print(f"sig len:  ",len(signal))
    print(f"time:     ",f"{str(len(signal)/fs)[:4]}s")
    print(f"size:      {str(len_binary_data/(8*1000))[:4]}KB")
    print(f"rate:      {str(len_binary_data*bits_per_value/(2*len(signal)))[:4]}")
    print(f"byterate:  {str(len_binary_data*fs/(8*1000*len(signal)))[:4]}KB/s")
    #print(f"LDPC:   n: {n}, ({d_v},{d_c})")
    print()

    gain = 1
    if save == True:
        ps.save_signal(signal,fs,f'test_signals/test_signal_{chirp_factor}c_{tracking_bins}t_{len(blocks)}b.wav')
    if play == True:
        ps.play_signal(signal*gain ,fs)
    
    







    