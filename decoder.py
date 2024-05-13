import csv
from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
from graycode import tc_to_gray_code as gray
import datetime



def signal_to_blocks(r_sig, N, pref_len):
    print("converting signal to blocks")

    r_sig_len = len(r_sig)
    print(r_sig_len,"datapoints")
    N_blocks = int(r_sig_len / (pref_len + N))
    print(N_blocks,"blocks")

    blocks = np.array_split(r_sig,N_blocks)

    blocks_trimmed =  np.array([block[pref_len:] for block in blocks])
    return blocks_trimmed



def channel_to_fft(channel, N, pref_len):
    print("performing channel fft")

    channel = np.pad(channel,(0,N - pref_len))
    channel_fft = np.fft.fft(channel)[0:int(N/2)]
    return channel_fft



def blocks_to_fft(blocks_trimmed, N):
    print("performing ffs on each block")

    blocks_fft = []
    for block in blocks_trimmed:
        block_fft = np.fft.fft(block)[0:int(N/2)]
        blocks_fft.append(list(block_fft))

    blocks_fft = np.array(blocks_fft)

    return blocks_fft



def divide_ffts(blocks_fft, channel_fft):
    print("adjusting for channel coefficients")

    blocks_adj_fft = []
    for block in blocks_fft:
        block_adj = block/channel_fft
        blocks_adj_fft.append(block_adj)

    return blocks_adj_fft



def blocks_to_bytes(blocks_adj_fft,M):
    print("starting decode...")

    blocks_phase = np.array([np.angle(block)[1:] for block in blocks_adj_fft])
    output = ""
    angle = 2*np.pi / M
    bits_per_symbol = int(np.log2(M))

    if M % 2 != 0:
        raise ValueError
    
    n = 0

    for block in blocks_phase:
        n += 1

        for phase in block:
            if phase < 0:
                phase = phase + 2*np.pi
            m = round((phase - angle/2) / angle) 
            format_graycode = f"0{str(bits_per_symbol)}b"
            graycode = format(gray(m),format_graycode)
            output += graycode

    output_bytes = []

    for n in range(len(output)//8):
        byte = output[8*n:8*(n+1)]
        byte = "0b" + byte
        byte = int(byte,0)
        output_bytes.append(byte)

    output_bytes = bytes(output_bytes)

    return output_bytes

def save_as_text(some_bytes,filename):
    output = str(some_bytes, 'utf-8',errors='ignore')
    with open(f"{filename}-text.txt","w") as output_file:
        output_file.write(output)
    output_file.close()

def save_as_hex(some_bytes,filename):
    output = ""
    for b in some_bytes:
        output += hex(b)
    with open(f"{filename}-hex.txt","w") as output_file:
        output_file.write(output)
    output_file.close()

def split_bytes(some_bytes):
    output = [[],[],[]]
    i=0
    bytes_list = list(some_bytes)
    for b in bytes_list:
        if b == 0 and i < len(output) - 1:
            i+=1
        else:
            output[i].append(b)
    filename = str(bytes(output[0]),encoding='utf8').split('/')[1]
    size = int(bytes(output[1]))
    data = bytes(output[2])

    print(f"Filename: {filename}, size: {size}")

    return filename, size, data


    





    








#1 232 tiff
#2 216 wav
#3 232 tiff
#4 tiff
#5
#6 216
#7 232
#8 
#9


# output = output[232:]
# with open("output.txt", "w") as text_file:
#     text_file.write(output)


# with open("parsed1.tiff", "wb") as text_file:
#     text_file.write(output)

# print("DONE")