
import numpy as np
from graycode import gray_code_to_tc
import visualize as v
import playsound as ps
import random
from matplotlib import pyplot as plt
from bitstring import BitArray
from py import ldpc

# with open('weekend-challenge/parsed.tiff',"rb") as file:
#      file_binary = file.read()

# binary = BitArray(file_binary).bin



### STANDARD ###
fs = 48000
block_length = 4096
prefix_length = 512 
N0 = 100
#N1 = 850
###
chirp_factor = 16
tracking_bins = 0
c = py.ldpc.code('802.16','3/4',81)
ldpc_factor = 1
###
used_bins = ( c.N// 2 ) * ldpc_factor
chirp_length = block_length * chirp_factor
used_bins_data = ( c.K // 2 ) * ldpc_factor
N1 = N0+ used_bins
###
play = False
save = True



def random_binary(N):
    np.random.seed(1)
    return np.random.randint(0,2,N)

def correct_binary_length(binary):
    number = used_bins_data*2 - (len(binary) % (used_bins_data*2))
    if number != used_bins_data*2:
        binary = np.pad(binary,(0,number))
    return binary

def encode_blocks(binary):
    if __name__ == "__main__":
        print("encoding blocks...",end="",flush=True)

    output = []
    
    for i in range((len(binary)//(used_bins_data*2))*ldpc_factor):
        block = binary[i*2*used_bins_data//ldpc_factor:(i+1)*2*used_bins_data//ldpc_factor]
        block = c.encode(block)
        output.extend(block)

    if __name__ == "__main__":
        print("done")
    return np.array(output)

def binary_to_values(binary):
    print("binary to values...",end="",flush=True)
    values = []
    for k in range(len(binary)//2):
        output = -1 + -1j
        value = binary[k*2:(k+1)*2]
        if value[0] == 0:
            output += 2j
        if value[1] == 0:
            output += 2
        values.append(output)
    print("done")
    return np.array(values)

def values_to_blocks(phases):
    print("values into blocks...",end="",flush=True)
    blocks = []
    for p in range(len(phases)//used_bins):
        block = phases[used_bins * p:used_bins*(1+p)]
        other_bins_factor = 0.5
        block = np.concatenate((other_bins_factor*np.exp(1j*(np.random.randint(0,4,N0)*np.pi/2 + np.pi/4)),block,other_bins_factor*np.exp(1j*(np.random.randint(0,4,block_length//2 + 1 - N0 - used_bins)*np.pi/2 + np.pi/4))))
        #block = np.pad(block,(N0,block_length//2 + 1- N0 - used_bins))
        # plt.scatter(list(range(len(block))),np.angle(block))
        # plt.show()
        blocks.append(block)
    print("done")
    return blocks

def blocks_fft_to_signal(blocks_fft,known_block_signal):
    print("ifft...",end="",flush=True)

    transmission = np.array([])
    for block in blocks_fft:
        block_signal = np.fft.irfft(block)
        block_signal /= np.sqrt(np.mean(np.absolute(block_signal)**2))
        block_signal = np.concatenate((block_signal[-prefix_length:],block_signal))
        transmission = np.concatenate((transmission,block_signal))
    # plt.plot(np.angle(block))
    # plt.show()
    
    chirp = ps.gen_chirp(N0,N0 + used_bins,fs,chirp_length,block_length)
    chirp = np.concatenate((chirp[-prefix_length:],chirp))
    chirp /= np.sqrt(np.mean(np.absolute(chirp)**2))
    transmission = np.concatenate((chirp,known_block_signal,transmission))
    print("done")
    transmission = transmission / np.max(transmission)
    return transmission

def generate_known_block(seed_=1):
    np.random.seed(seed_)
    graycode = np.random.randint(0,4,block_length//2 - 1)
    values = graycode * (np.pi/2) + np.pi/4
    for i in range(len(values)):
        value = values[i]
        if value > np.pi:
            values[i] = -(2*np.pi - value)
    values = np.exp(1j*values)
    
    values = np.concatenate(([0],values,[0]))
    #v.plot_fft(values,fs)
    known_block_signal = np.fft.irfft(values)
    known_block_signal = np.concatenate((known_block_signal[-prefix_length:],known_block_signal))

    return known_block_signal/np.sqrt(np.mean(np.absolute(known_block_signal)**2))

if __name__ == "__main__":
    
    binary = random_binary(used_bins_data*100*2)
    len_binary_data = len(binary)
    binary = correct_binary_length(binary)
    binary = encode_blocks(binary)
    values = binary_to_values(binary)/np.sqrt(2)
    blocks_fft = values_to_blocks(values)
    known_block_signal = generate_known_block()
    signal = blocks_fft_to_signal(blocks_fft,known_block_signal)
    plt.plot(signal)
    plt.show()
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
        ps.save_signal(signal,fs,f'test_signals/test_signal_{c.standard}_{c.N}_{c.K}_{N0}_{N1}.wav')
    if play == True:
        ps.play_signal(signal*gain ,fs)
    
    







    