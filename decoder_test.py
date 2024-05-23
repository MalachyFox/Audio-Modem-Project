import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
import matplotlib.pyplot as plt
import decoder as d
import encoder as e

def get_fft_chirp(chirp,overlap = False):
    fft_chirp = np.zeros(block_length//2,dtype=np.complex_)
    if overlap == True:
        for i in range(chirp_factor* 2 - 1):
            partial_chirp = chirp[i*block_length//2:(i+2)*block_length//2]
            fft_partial_chirp = np.fft.rfft(partial_chirp)[:-1]
            fft_chirp += fft_partial_chirp
            #visualize.plot_fft(fft_chirp,fs)
    else:
        for i in range(chirp_factor):
            partial_chirp = chirp[i*block_length:(i+1)*block_length]
            fft_partial_chirp = np.fft.rfft(partial_chirp)[:-1]
            fft_chirp += fft_partial_chirp

    return fft_chirp







### STANDARD ###
fs = 48000
block_length = 4096 
bits_per_value = 2
prefix_length = 512 
N0 = 85
N1 = 850
###
recording_time = 14
chirp_factor = 16
tracking_length = 20
num_blocks = 100
###
used_bins = N1 - N0
chirp_length = block_length * chirp_factor
used_bins_data = used_bins - tracking_length
###
record = False
use_test_signal = False



### sync function ###
sync_chirp = playsound.gen_chirp(N0,N1,fs,chirp_length,block_length)
sync = np.concatenate((sync_chirp[-prefix_length:],sync_chirp))



### start recording ###
if record == True:
    input = input('press enter')
    recording = sd.rec(fs * recording_time,samplerate = fs,channels=1)
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
position = np.argmax(correlation) + 1# +1 moves slopes upwards CCW

# plt.plot(correlation)
# plt.show()

### estimate channel ###
chirp = recording[position - len_sync_chirp :position]
#chirp *= scipy.signal.windows.hamming(block_length)
fft_chirp = get_fft_chirp(chirp)

fft_sync_chirp = get_fft_chirp(sync_chirp)

channel = fft_chirp/fft_sync_chirp

channel = channel[N0:N1] # maybe not useful
channel = np.concatenate((np.ones(N0),channel,np.ones(block_length//2- N1)))
impulse = np.fft.irfft(channel)
#visualize.plot_channel(impulse)



### reverse channel effects ###

blocks = []
block_index = 0
order = 2
coefs = np.zeros(order)
group_length = prefix_length + block_length
while True:
    start = position + prefix_length + group_length * block_index
    end = position + group_length + group_length * block_index
    data = recording[start:end]

    data_fft = np.fft.rfft(data)
    data_fft = data_fft[N0:N1]
    data_fft = data_fft/(channel[N0:N1])

    for k in range(len(data_fft)):
        f =  N0 + k
        angle = np.exp(-1j*(np.sum([a*f**b for a,b in zip(coefs,np.flip(list(range(order))))]))) # ignore this unholy one liner to do polynomials
        data_fft[k] = data_fft[k] * angle

    spacing = used_bins//(tracking_length)
    pilot_indices = np.array([i*spacing for i in range(tracking_length)])
    pilots = np.array([np.angle(data_fft[i]) for i in pilot_indices]) - np.pi/4
    freqs = pilot_indices + N0
    coefs_new = np.polyfit(freqs,pilots,order - 1)
    #print(coefs_new)
    for k in range(len(data_fft)):
        f =  N0 + k
        angle = np.exp(-1j*np.sum(([a*f**b for a,b in zip(coefs_new,np.flip(list(range(order))))])))
        data_fft[k] = data_fft[k] * angle

    coefs += coefs_new
    blocks.append(data_fft)
    block_index += 1
    if block_index == num_blocks:
        break


### decode signal ###
bytes_list, r_bits = d.blocks_to_bytes(blocks,4)
t_bits = e.random_binary(used_bins_data*bits_per_value*num_blocks)
t_bits = e.add_tracking(t_bits)
t_bits = e.correct_binary_length(t_bits)



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
    print(" rec:",r[:view],"...",r[-view:])
    print("sent:",t[:view],"...",t[-view:])
    print()
print(f"TOTAL ERRORS: {(str(total_errors/num_blocks))[:4]}%")

# plt.plot(error_list)
# plt.show()



### view plots ###
visualize.big_plot(blocks[:10],fs,title="test",colours=colours)

full_range = []
for b in blocks:
    full_range = np.concatenate((full_range,b))


#visualize.plot_fft(full_range,fs,colours)
visualize.plot_constellation(full_range,colours=colours)

