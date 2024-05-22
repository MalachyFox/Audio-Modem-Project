from ctypes import c_buffer
from re import I
from matplotlib.mlab import phase_spectrum
import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
import matplotlib.pyplot as plt
import decoder as d
import encoder as e

seconds = 8
fs = 44100
gain = 1
f0 = 500
block_length = 10000
f1 = f0 + block_length
num_blocks = 4
record = False

#generate double sync function
sync_chirp = playsound.gen_chirp(f0,f1,fs,2*f1/fs)
sync = np.concatenate((sync_chirp,sync_chirp,sync_chirp))


#start recording
if record == True:
    input = input('press space')
    recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
    sd.wait()
    recording = recording.flatten()
    playsound.save_signal(recording,fs,f'recordings/recording_{f0}_{f1}_{num_blocks}b.csv')
else:
    recording = playsound.load_signal(f'recordings/recording_{f0}_{f1}_{num_blocks}b.csv') #(f'test_signals/test_signal_{f0}_{f1}_{fs}_{num_blocks}b.wav')   #(f'recordings/recording_{f0}_{f1}_{num_blocks}b.csv') #
    recording = recording.flatten()


# find position
len_sync_chirp = len(sync_chirp)
correlation = scipy.signal.correlate(recording, sync)
position_data = np.argmax(correlation)
position = position_data - len_sync_chirp*2 # start of 1st chirp (no prefix)
plt.plot(correlation)
plt.show()

def CFO(sync):
    chirp1 = recording[position : position + len(sync)//2]
    chirp2 = recording[position + len(sync)//2 :position + len(sync)]
    fft_chirp1 = np.fft.fft(chirp1)[f0:f1]
    fft_chirp2 = np.fft.fft(chirp2)[f0:f1]

    #Calculate the phase shift between chirp1 and 2
    phase_diff_CFO = np.angle(fft_chirp2 * np.conj(fft_chirp1))
    phase_diff_CFO = np.mean(phase_diff_CFO)
    #CFO_correction = np.exp(-2j * CFO(sync))
    #CFO_correction = np.exp(-1j * estimate_phase_offset(recording, sync))
    #data_fft = data_fft * CFO_correction
    #sfo = 0
    # for i in range(f0, f1):
    #     sfo += i * np.angle((fft_chirp2[i]) * np.conj(fft_chirp1[i])) 
    # sfo = sfo / sum([i**2 for i in range(f0,f1)])

    # print('SFO:', sfo)

    return phase_diff_CFO

# plt.plot(recording)
# plt.show()
# estimate channel
chirp1 = recording[position : position + f1*2]
chirp2 = recording[position + len_sync_chirp :position+len_sync_chirp + f1*2]



fft_chirp1 = np.fft.rfft(chirp1)
fft_chirp2 = np.fft.rfft(chirp2)

chirp_adjust = fft_chirp2/fft_chirp1
#visualize.plot_fft(fft_chirp1,fs)
#visualize.plot_fft(fft_chirp2,fs)

fft_sync_chirp = np.fft.rfft(sync_chirp[:f1*2])
#visualize.plot_fft(fft_sync_chirp,fs)

channel_raw = fft_chirp2 / fft_sync_chirp
channel_chop = channel_raw[f0:f1]
channel = np.concatenate((np.zeros(f0),channel_chop,np.zeros(1)))
impulse = np.fft.irfft(channel)
visualize.plot_channel(impulse)

#perform least squares on the two chirps
x = np.linspace(f0,f1 - 1,f1-f0)
y = np.angle(fft_chirp2[f0:f1] * np.conj(fft_chirp1[f0:f1]))
m_, c_ = np.polyfit(x,y,1)

resid = np.array([abs((y - (m_*x + c_)))**8 for x, y in zip(x,y)])
print(resid)
m, c = np.polyfit(x,y,1,w=1/resid)

plt.scatter(x,y,alpha=0.1)
plt.plot(x,x*m_ + c_,label="1",c="b")
plt.plot(x,x*m + c,label="2",c="r")
plt.legend()
plt.show()



# reverse channel effects
prefix_samples = f1*2
block_samples = f1*2

blocks = []
i=0
while True:

    group_length = prefix_samples + block_samples
    start = position_data + prefix_samples + group_length * i
    end = position_data + group_length + group_length * i
    data = recording[start:end]
    data_fft = np.fft.rfft(data)
    data_fft = data_fft[f0:f1]
    data_fft = data_fft/(channel[f0:f1])
    
    #data_fft *= np.exp(-1j * np.angle(chirp_adjust[f0:f1])*(i+1)*4)
    #make cfo and sfo adjustment

    bm = 0#4*(i+1)
    bc = 0#2    #1

    for k in range(len(data_fft)):
        f =  f0 + k
        angle = np.exp(-1j*(m*f*bm + c*bc))
        data_fft[k] = data_fft[k] * angle

    blocks.append(data_fft)
    i += 1
    if i == num_blocks:
        break


#decode signal
bytes_list, r_bits = d.blocks_to_bytes(blocks,4)

t_bits = e.random_binary(block_length*2*num_blocks)


#add colours
colours = []
for i in range(len(t_bits)//2):
    bit = t_bits[i*2:(i+1)*2]
    if bit == "00":
        colours.append("r")
    elif bit == "01":
        colours.append("y")
    elif bit == "11":
        colours.append("g")
    elif bit == "10":
        colours.append("b")

#compare signals
total_errors = 0
for b in range(len(blocks)):
    r = r_bits[b*block_length*2:(b+1)*block_length*2]
    t = t_bits[b*block_length*2:(b+1)*block_length*2]
    count = sum(1 for a,b in zip(r,t)if a != b) /(block_length*2) * 100
    total_errors += count
    errors = str(count)[:4] + "%"
    print(f"block {b}, {errors} errors")
    view = 20
    print(" rec:",r[:view],"...",r[-view:])
    print("sent:",t[:view],"...",t[-view:])
    print()

print(f"TOTAL ERRORS: {(str(total_errors/num_blocks))[:4]}%")


visualize.big_plot(blocks,fs,colours,title="test")

individual = False
if individual == True:
    for i in range(num_blocks):
        col = colours[i*block_length:(i+1)*block_length]
        visualize.plot_fft(blocks[i],fs,f0,f1,title=f"{errors}")
        visualize.plot_constellation(blocks[i],col,title=f"{errors}")

