from uu import encode
import bitarray
import numpy as np
from graycode import gray_code_to_tc
import visualize as v
import playsound as ps
import random
from matplotlib import pyplot as plt
from bitstring import BitArray
import ldpc

# with open('weekend-challenge/parsed.tiff',"rb") as file:
#      file_binary = file.read()

# binary = BitArray(file_binary).bin

fs = 44100
M = 4
m = int(np.log2(M))
block_length = 16384
data_block_length = block_length
f0 = 1000
f1 = f0 + block_length
n=12
d_v = 3
d_c = 6

def random_binary(N):
    random.seed(1)
    return ''.join(random.choices(["0","1"], k=N))

def correct_binary_length(binary,m=m):
    one = int(len(binary)%m)
    binary = binary +  (m - one)%m * "0" # makes sure binary can be divided into values
    two = len(binary)//m%data_block_length
    binary = binary +  (data_block_length - two)%(data_block_length) * m * "0" # makes sure binary can be divided into blocks
    return binary

def binary_str_to_symbols(binary,m=m):
    symbols = []
    for i in range(len(binary)//m):
        symbols.append(binary[m*i:m*(i+1)])
    return symbols

def symbols_to_phases(symbols,M=M):
    phases = []
    for symbol in symbols:

        b_int = gray_code_to_tc(int(symbol,2))
        b_phase = (b_int + 0.5)*2*np.pi /M    #b_int + 0.5? ??
        if b_phase > np.pi:
            b_phase = -(2*np.pi - b_phase)
        phases.append(b_phase)
    #print(b_int,value,b_phase)
    return phases

def phases_to_blocks(phases,data_block_length=data_block_length):
    blocks = []
    for p in range(int(len(phases)/data_block_length)):
        blocks.append(phases[data_block_length * p:data_block_length*(1+p)])
    return blocks

def blocks_to_blocks_fft(blocks,fs=fs,f0=f0,f1=f1):
    f_d_half = np.zeros(int(fs/2),dtype=np.complex_)
    blocks_fft = []
    for block in blocks:
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
    return blocks_fft

def blocks_fft_to_signal(blocks_fft,fs=fs,f0=f0,f1=f1):
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
        
        transmission = np.concatenate((transmission,signal))
    chirp = ps.gen_chirp(f0,f1,fs,1)
    chirp2 = ps.double_signal(chirp)
    transmission = np.concatenate((chirp2,transmission))
    return transmission

def prep_ldpc_encode(binary,n=n,d_v=d_v,d_c=d_c):

    H, G = ldpc.make_ldpc(n, d_v, d_c,systematic=True,seed=1)

    len_msg = G.shape[1] # v # length of message
    #print(len_msg)
    messages = []

    # round to message lengths
    if len(binary) % len_msg >0:
        binary += (len(binary)%len_msg)*"0" 
    
    #convert to array of integers
    for i in range(len(binary)//len_msg):
        message = binary[i*len_msg:(i+1)*len_msg]
        message_ = []
        for b in message:
            message_.append(int(b))
        messages.append(np.array(message_))


    
    #print(G)
    #encode
    encoded_messages = []
    for message in messages:
        encoded_messages.append(ldpc.ldpc_encode(G,message=message))

    #convert to string of bits
    output = ""
    for encoded_messsage in encoded_messages:
        for bit in encoded_messsage:
            output += str(bit)
    return output

if __name__ == "__main__":

    binary = random_binary(block_length*m)
    binary = "00000000111111110101010110101010" + binary[:-33]
    #binary = prep_ldpc_encode(binary)
    binary = correct_binary_length(binary)
    symbols = binary_str_to_symbols(binary)
    phases = symbols_to_phases(symbols)
    blocks = phases_to_blocks(phases)
    blocks_fft = blocks_to_blocks_fft(blocks)
    signal = blocks_fft_to_signal(blocks_fft)

    print("fs:",fs,'f0:',f0,'f1:',f1,"M:",M,"block length:",block_length, 'n:',n,'d_v:',d_v,'d_c:',d_c)

    print("BINARY:",binary[:100],"...")
    print("NUMBER OF BLOCKS:", len(blocks))

    gain = 1
    ps.play_signal(signal*gain ,fs)










    