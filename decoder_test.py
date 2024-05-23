import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
import matplotlib.pyplot as plt
import decoder as d
import encoder as e

def get_fft_chirp(chirp):
    fft_chirp = np.zeros(block_length//2,dtype=np.complex_)
    for i in range(chirp_factor):
        partial_chirp = chirp[i*block_length:(i+1)*block_length]
        fft_partial_chirp = np.fft.rfft(partial_chirp)[:-1]
        fft_chirp += fft_partial_chirp
    return fft_chirp





seconds = 5
fs = 48000
block_length = 4096 # 10000
chirp_factor = 16
chirp_length = block_length * chirp_factor
tracking_length = 4 # 100
prefix_length = 512 # 500
N0 = 85
N1 = 850
used_bins = N1 - N0
used_bins_data = used_bins - tracking_length*2

num_blocks = 4

bits_per_value =2
gain = 1

record = False
use_test_signal = True



### sync function ###
sync_chirp = playsound.gen_chirp(N0,N1,fs,chirp_length)
sync = np.concatenate((sync_chirp[-prefix_length:],sync_chirp))



### start recording ###
if record == True:
    input = input('press enter')
    recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
    sd.wait()
    recording = recording.flatten()
    playsound.save_signal(recording,fs,f'recordings/recording_{chirp_factor}c_{tracking_length}t_{num_blocks}b.csv')
else:
    if (use_test_signal):
        recording = playsound.load_signal(f'test_signals/test_signal_{chirp_factor}c_{tracking_length}t_{num_blocks}b.wav')
    else:
        recording = playsound.load_signal(f'recordings/recording_{chirp_factor}c_{tracking_length}t_{num_blocks}b.csv') #   #(f'recordings/recording_{f0}_{f1}_{num_blocks}b.csv') #
    
    recording = recording.flatten()
print("done recording")


### find position ###
len_sync_chirp = len(sync_chirp)
correlation = scipy.signal.correlate(recording, sync)
position = np.argmax(correlation) +1 # +1 moves slopes upwards CCW

plt.plot(correlation)
plt.show()

### estimate channel ###
chirp = recording[position - len_sync_chirp :position]
#chirp *= scipy.signal.windows.hamming(block_length)
fft_chirp = get_fft_chirp(chirp)

fft_sync_chirp = get_fft_chirp(sync_chirp)

channel = fft_chirp/fft_sync_chirp
channel = channel[N0:N1] # maybe not useful
channel = np.concatenate((np.ones(N0),channel,np.ones(block_length//2- N1)))
impulse = np.fft.irfft(channel)
visualize.plot_channel(impulse)



### reverse channel effects ###

blocks = []
i=0
m=0
c=0
group_length = prefix_length + block_length
while True:
    start = position + prefix_length + group_length * i
    end = position + group_length + group_length * i
    data = recording[start:end]

    data_fft = np.fft.rfft(data)
    data_fft = data_fft[N0:N1]
    data_fft = data_fft/(channel[N0:N1])

    # for k in range(len(data_fft)):
    #     f =  f0 + k
    #     angle = np.exp(-1j*(m*f + c))
    #     data_fft[k] = data_fft[k] * angle

    # start = np.mean(np.angle(data_fft[:tracking_length]))
    # end = np.mean(np.angle(data_fft[-tracking_length:]))
    # m2 = (end -start) / (f1-f0-tracking_length)
    # c2 = 0 - m2*(f0+tracking_length//2)

    # for k in range(len(data_fft)):
    #     f =  f0 + k
    #     angle = np.exp(-1j*(m2*f + c2))
    #     data_fft[k] = data_fft[k] * angle

    # m+= m2
    # c+= c2
    blocks.append(data_fft)
    i += 1
    if i == num_blocks:
        break


### decode signal ###
bytes_list, r_bits = d.blocks_to_bytes(blocks,4)
t_bits = e.random_binary(used_bins_data*2*num_blocks)
t_bits = e.add_tracking(t_bits)



### add colours ###         # not scalable for M-ary yet
colours = []
for i in range(len(t_bits)//bits_per_value):
    bit = t_bits[i*bits_per_value:(i+1)*bits_per_value]
    if bit == "00":
        colours.append("r")
    elif bit == "01":
        colours.append("y")
    elif bit == "11":
        colours.append("g")
    elif bit == "10":
        colours.append("b")



### compare signals ###
total_errors = 0
error_list = []
for b in range(len(blocks)):
    r = r_bits[b*used_bins*2:(b+1)*used_bins*2]
    t = t_bits[b*used_bins*2:(b+1)*used_bins*2]
    count = sum(1 for a,b in zip(r,t)if a != b) /(used_bins*bits_per_value) * 100
    error_list.append(count)
    total_errors += count
    errors = str(count)[:4] + "%"
    print(f"block {b}, {errors} errors")
    view = 20
    #print(" rec:",r[:view],"...",r[-view:])
    #print("sent:",t[:view],"...",t[-view:])
    #print()
print(f"TOTAL ERRORS: {(str(total_errors/num_blocks))[:4]}%")

plt.plot(error_list)
plt.show()
### view plots ###
visualize.big_plot(blocks,fs,colours,title="test")

individual = False
if individual == True:
    for i in range(num_blocks):
        col = colours[i*block_length:(i+1)*block_length]
        visualize.plot_fft(blocks[i],fs,f0,f1,title=f"{errors}")
        visualize.plot_constellation(blocks[i],col,title=f"{errors}")

