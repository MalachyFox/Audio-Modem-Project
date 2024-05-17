import bitarray
import numpy as np
from graycode import gray_code_to_tc
import visualize as v
import playsound as ps
import random
from matplotlib import pyplot as plt
from bitstring import BitArray

def random_binary(N):
    random.seed(1)
    return ''.join(random.choices(["0","1"], k=N))



# with open('weekend-challenge/parsed.tiff',"rb") as file:
#      file_binary = file.read()

# binary = BitArray(file_binary).bin

M = 4
m = int(np.log2(M))
block_length = 2048
#print(binary[:20])
binary = random_binary(block_length*m)
binary = "00000000111111110101010110101010" + binary[:-33]
#binary = "00011110"*125*2
print(binary[:100])

data_block_length = block_length #//2 # int(block_length /2 - 1)
fs = 44100


one = int(len(binary)%m)
binary = binary +  (m - one)%m * "0" # makes sure binary can be divided into values
two = len(binary)//m%data_block_length
binary = binary +  (data_block_length - two)%(data_block_length)  *m* "0" # makes sure binary can be divided into blocks

binary_list = []
for i in range(int(len(binary)//m)):
    binary_list.append(binary[m*i:m*(i+1)])

phase_list = []
for value in binary_list:
    #print(value)
    b_int = gray_code_to_tc(int(value,2))
    
    b_phase = (b_int + 0.5)*2*np.pi /M    #b_int + 0.5? ??
    if b_phase > np.pi:
        b_phase = -(2*np.pi - b_phase)
    phase_list.append(b_phase)
    #print(b_int,value,b_phase)


blocks_list = []
for p in range(int(len(phase_list)/data_block_length)):
    blocks_list.append(phase_list[data_block_length * p:data_block_length*(1+p)])

print("NUMBER OF BLOCKS:", len(blocks_list))

f0 = 1000
f1 = f0 + block_length

f_d_half = np.zeros(int(fs/2),dtype=np.complex_)

blocks_fft = []
for block in blocks_list:
    block_f_d = f_d_half
    #print(len(block), "LENGTH BLOCK")
    for i in range(int(len(block))):
        block_f_d[f0 + i] = 1 * np.cos(block[i]) + 1j * np.sin(block[i])
        #block_f_d[f1 - i - 1] = 1 * np.cos(block[i]) - 1j * np.sin(block[i])
    #ft = np.concatenate((block_f_d,np.flip(np.conjugate(block_f_d))))
    ft = block_f_d
    blocks_fft.append(ft)
    #plt.plot(np.absolute(ft))
    #plt.show()
    #v.plot_fft(ft,fs,0,fs//2)

chirp = ps.gen_chirp(f0,f1,fs,1)
chirp = ps.double_signal(chirp)


transmission = []
for block in blocks_fft:
    signal = np.fft.irfft(block,fs)
    #print(len(signal),"LENGTH")
    dur = len(signal)/fs
    max = np.max(signal)
    signal = signal /max # normalise to avoid clipping
    #signal += ps.gen_sine(f0 - 1,fs,dur)# + freqs//2,fs,dur)
    signal = np.concatenate((signal,signal))

    #plt.plot(signal)
    
    #v.plot_fft(np.fft.rfft(signal),fs)
    signal = np.concatenate((chirp,signal))
    transmission = np.concatenate((transmission,signal))
ps.play_signal(transmission*1 ,fs)










    