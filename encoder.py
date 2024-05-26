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
import py.ldpc

# with open('weekend-challenge/parsed.tiff',"rb") as file:
#      file_binary = file.read()

# binary = BitArray(file_binary).bin



### STANDARD ###
fs = 48000
block_length = 4096
prefix_length = 512 
N0 = 85
N1 = 850
###
chirp_factor = 16
tracking_bins = 0
num_blocks = 100
c = py.ldpc.code('802.11n','3/4',54)
ldpc_factor = 1
###
used_bins = ( c.N// 2 ) * ldpc_factor
chirp_length = block_length * chirp_factor
used_bins_data = ( c.K // 2 ) * ldpc_factor
###
play = False
save = True



def random_binary(N):
    random.seed(1)
    return np.array(random.choices([0,1], k=N))

def correct_binary_length(binary):
    one = len(binary)%2
    binary = binary +  (2 - one)%2 * "0" # makes sure binary can be divided into values
    two = len(binary)//2%used_bins
    binary = binary +  (used_bins - two)%(used_bins) * 2 * "0" # makes sure binary can be divided into blocks
    return binary

def values_to_blocks(phases):
    blocks = []
    for p in range(len(phases)//used_bins):
        block = phases[used_bins * p:used_bins*(1+p)]
        block = np.pad(block,(N0,block_length//2 + 1- N0 - used_bins))
        blocks.append(block)
    return blocks

def blocks_fft_to_signal(blocks_fft):

    transmission = np.array([])
    for block in blocks_fft:
        block_symbol = np.fft.irfft(block)
        block_symbol = np.concatenate((block_symbol[-prefix_length:],block_symbol))
        transmission = np.concatenate((transmission,block_symbol))

    transmission = transmission /np.max(transmission)
    chirp = ps.gen_chirp(N0,N0 + used_bins,fs,chirp_length,block_length)
    chirp = np.concatenate((chirp[-prefix_length:],chirp))
    chirp = chirp/np.max(chirp)
    transmission = np.concatenate((chirp,transmission))
    
    return transmission

def binary_to_values(binary):
    values = []
    for k in range(len(binary)//2):
        output = -1 + -1j
        value = binary[k*2:(k+1)*2]
        if value[0] == 0:
            output += 2j
        if value[1] == 0:
            output += 2
        values.append(output)
    return np.array(values)/np.sqrt(2)

def encode_blocks(binary,num_blocks=num_blocks):
    output = []
    
    for i in range(num_blocks*ldpc_factor):
        block = binary[i*2*used_bins_data//ldpc_factor:(i+1)*2*used_bins_data//ldpc_factor]
        block = c.encode(block)
        output.extend(block)
    return np.array(output)



if __name__ == "__main__":
    
    binary = random_binary(used_bins_data*2*num_blocks)
    len_binary_data = len(binary)
    binary = encode_blocks(binary)
    values = binary_to_values(binary)
    blocks_fft = values_to_blocks(values)
    signal = blocks_fft_to_signal(blocks_fft)

    print()
    print(f"fs:       ",fs)
    print(f'N0:       ',N0)
    print(f'N1:       ',N0 + used_bins)
    print(f"blck len: ",block_length)
    print(f"prfx len: ",prefix_length)
    print(f'ldpc:      {c.standard} {c.K/c.N} {c.z}')
    print()
    print(f"num blcks:",len(blocks_fft))
    print(f"time:     ",f"{str(len(signal)/fs)[:4]}s")
    print(f"size:      {str(len_binary_data/(8*1000))[:4]}KB")
    print(f"rate:      {str(len_binary_data*2/(2*len(signal)))[:4]}")
    print(f"byterate:  {str(len_binary_data*fs/(8*1000*len(signal)))[:4]}KB/s")
    print()

    gain = 1
    if save == True:
        ps.save_signal(signal,fs,f'test_signals/test_signal_{c.standard}_{c.N}_{c.K}_{len(blocks_fft)}b.wav')
    if play == True:
        ps.play_signal(signal*gain ,fs)
    
    







    