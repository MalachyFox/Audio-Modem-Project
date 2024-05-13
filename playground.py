import decoder as d
import numpy as np
import visualize as v

pref_len = 32
N = 1024
M = 4

r_sig = np.genfromtxt('weekend-challenge/file1.csv')
channel = np.genfromtxt('weekend-challenge/channel.csv')
v.plot_channel(channel)
blocks_trimmed = d.signal_to_blocks(r_sig,N,pref_len)
channel_fft = d.channel_to_fft(channel, N, pref_len)
blocks_fft = d.blocks_to_fft(blocks_trimmed, N)

blocks_fft_adj = d.divide_ffts(blocks_fft,channel_fft)

v.plot_fft(blocks_fft_adj[2])
v.plot_constellation(blocks_fft_adj[2])


output = d.blocks_to_bytes(blocks_fft_adj,M)
filename, size, data = d.split_bytes(output)

d.save_as_text(data,filename)
d.save_as_hex(data,filename)

with open(filename,"wb") as output_file:
    output_file.write(data)
output_file.close()

print("DONE")



