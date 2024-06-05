
import numpy as np
from graycode import gray_code_to_tc
import visualize as v
import playsound as ps
import random
from matplotlib import pyplot as plt
from bitstring import BitArray
from py import ldpc




### STANDARD ###
fs = 48000
block_length = 4096
prefix_length = 1024 
B0 = 85
#N1 = 850
###
chirp_factor = 16
c = ldpc.code('802.16','1/2',54)
ldpc_factor = 1
###
used_bins = ( c.N// 2 ) * ldpc_factor
chirp_length = block_length * chirp_factor
used_bins_data = ( c.K // 2 ) * ldpc_factor
B1 = B0 + used_bins
###
play = True
save = True

num_known_block = 1


def random_binary(N):
    np.random.seed(1)
    return np.random.randint(0,2,N)

def correct_binary_length(binary):
    number = used_bins_data*2 - (len(binary) % (used_bins_data*2))
    if number != used_bins_data*2:
        binary = np.concatenate((binary,np.random.randint(0,2,number)))
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
    return np.array(values)/np.sqrt(2)

def values_to_blocks(phases,known_block):
    print("values into blocks...",end="",flush=True)
    blocks = []
    for p in range(len(phases)//used_bins):
        block = phases[used_bins * p:used_bins*(1+p)]
        block = np.concatenate((np.ones(B0),block,np.ones(block_length//2 + 1 - B0 - used_bins)))
        block *= known_block/np.exp(1j*np.pi/4)
        blocks.append(block)
    print("done")
    return blocks

def blocks_fft_to_signal(blocks_fft,known_block_signal):
    print("ifft...",end="",flush=True)
    

    transmission = np.array([])
    for block in blocks_fft:
        block_signal = np.fft.irfft(block)
        #block_signal /= np.sqrt(np.mean(np.absolute(block_signal)**2))
        block_signal = np.concatenate((block_signal[-prefix_length:],block_signal))
        transmission = np.concatenate((transmission,block_signal))
    known_block_signal = np.tile(known_block_signal,num_known_block) ### REMOVE LINE FOR 1 KNOWN BLOCK
    transmission = np.concatenate((known_block_signal,transmission))
    transmission = transmission / np.max(transmission)
    chirp = ps.gen_chirp(B0,B0 + used_bins,fs,chirp_length,block_length)
    chirp = np.concatenate((chirp[-prefix_length:],chirp,chirp[:prefix_length]))

    chirp /= np.max(np.absolute(chirp))
    chirp*=0.1
    transmission = np.concatenate((chirp,transmission,chirp))
    transmission /= np.max(np.absolute(transmission))
    print("done")
    
    return transmission

def generate_known_block():
    np.random.seed(1)
    graycode = np.random.randint(0,4,block_length//2 - 1)
    values = graycode * (np.pi/2) + np.pi/4
    for i in range(len(values)):
        value = values[i]
        if value > np.pi:
            values[i] = -(2*np.pi - value)
    values = np.exp(1j*values)

    ##old coding method
    # values = []
    # for g in graycode:
    #     output = -1 + -1j
    #     value =f"{g:02b}"
    #     if int(value[0]) == 0:
    #         output += 2j
    #     if int(value[1]) == 0:
    #         output += 2
    #     values.append(output)
    #print(len(values))

    values = np.concatenate(([0],values,[0]))
    known_block_signal = np.fft.irfft(values)
    known_block_signal = np.concatenate((known_block_signal[-prefix_length:],known_block_signal))

    return known_block_signal, values #/np.sqrt(np.mean(np.absolute(known_block_signal)**2))

def load_file(filename):
    ### load file
    with open("./sendable_files/" + filename,"rb") as file:
        file_binary = file.read()
    binary = np.array([int(b) for b in BitArray(file_binary).bin])
    return binary
    
def add_header(binary,filename):
    ### prepare header
    filesize = len(binary)
    head = "\0\0" + filename + "\0\0" + str(filesize) + "\0\0"
    head = bytearray(head,"utf8")
    head_old = [f'{int(bin(byte)[2:]):08d}' for byte in head]
    head= []
    for byte in head_old:
        for bit in byte:
            head.append(int(bit))
    head = np.array(head)
    binary = np.concatenate((head,binary))
    return binary

def handle_header(binary):
    bytes_list = []
    for i in range(len(binary)//8):
        byte = binary[i*8:(i+1)*8]
        #print(byte)
        byte_str = ""
        for b in byte:
            byte_str += str(b)
        byte_str = "0b" + byte_str
        bytes_list.append(int(byte_str,0))
    bytes_list = np.array(bytes_list,dtype=np.uint8)
    [print(chr(a),end='') for a in bytes_list[:40]]
    print()
    print(bytes_list[:20])
    #print(len(bytes_list)/1000)
    inds = np.where(bytes_list == 0)[0]
    filename_temp = bytes_list[inds[1] + 1:inds[2]]
    filename = ""
    for h in filename_temp:
         filename += chr(h)

    size = ""
    size_temp = bytes_list[inds[3] + 1:inds[4]]
    for s in size_temp:
         size += chr(s)
    
    size = int(size)

    data = bytes_list[inds[5] +1:]
    data = bytes(data[:size//8])
    #print(len(data)/1000)
    # print(bytes_list[:20])
    # inds = np.where(bytes_list == 0)[0]
    # filename_temp = bytes_list[inds[0] + 1:inds[1]]
    # filename = ""
    # for h in filename_temp:
    #      filename += chr(h)

    # size = ""
    # size_temp = bytes_list[inds[1] + 1:inds[2]]
    # for s in size_temp:
    #      size += chr(s)
    # size = int(size)

    # data = bytes_list[inds[3] +1:]
    # data = bytes(data[:size//8])


    
    return filename, size, data

    

if __name__ == "__main__":
    
    filename = 'hamlet.txt'  # in sendable_files/
    binary = load_file(filename)
    binary = add_header(binary,filename)
    len_binary_data = len(binary)
    binary = correct_binary_length(binary)
    binary = encode_blocks(binary)
    values = binary_to_values(binary)
    known_block_signal, known_block_fft = generate_known_block()
    blocks_fft = values_to_blocks(values,known_block_fft)
    signal = blocks_fft_to_signal(blocks_fft,known_block_signal)



    print()
    print(f"fs:         {fs}Hz")
    print(f'B0:        ',B0)
    print(f'B1:        ',B0 + used_bins)
    print(f"block len: ",block_length)
    print(f"prefix len:",prefix_length)
    print(f"chirp len:  {chirp_factor} x {block_length}")
    print(f"chirp frqs: {(B0-20)*fs/block_length}Hz -> {(B1+20)*fs/block_length}Hz")
    print(f'LDPC:       {c.standard}, {c.K/c.N}, {c.z}')
    print()
    print(f"num blocks: {len(blocks_fft)} + 5 known block2")
    print(f"time:       {str(len(signal)/fs)[:4]}s")
    print(f"size:       {len(blocks_fft)*used_bins_data*2/8000:.8g}kB")
    print(f"size:       {len_binary_data/(8*1000):.4g}kB")
    print(f"efficiency: {str(len_binary_data*2/(2*len(signal)))[:4]}")
    print(f"byte-rate:  {str(len_binary_data*fs/(8*1000*len(signal)))[:4]}kB/s")
    print()

    if save == True:
        ps.save_signal(signal,fs,f'test_signals/{filename}.wav')
        #ps.save_signal(signal,fs,f'test_signals/test_signal_{c.standard}_{c.N}_{c.K}_{B0}_{B1}.wav')
    
    if play == True:
        ps.play_signal(signal ,fs)

    # plt.plot(signal)
    # plt.show()
    
    







    