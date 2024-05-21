from click import group
from matplotlib.mlab import phase_spectrum
import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
from ctypes.util import find_library
import matplotlib.pyplot as plt
import decoder as d
import encoder as e
find_library('portaudio')

seconds = 14
fs = 44100
gain = 2
f0 = 1000
block_length = 2048
f1 = f0 + block_length
num_blocks = 4

#generate double sync function
sync_chirp = playsound.gen_chirp(f0,f1,fs,1)
sync = np.concatenate((sync_chirp,sync_chirp,sync_chirp))


#start recording
input = input('press space')
recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()
recording = recording.flatten()


# find position
correlation = scipy.signal.correlate(recording, sync)
peak_correlation = np.max(correlation)
position_data = np.argmax(correlation)
position = position_data - len(sync_chirp)*2 # start of 1st chirp (no prefix)

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


# estimate channel
chirp1 = recording[position : position + len(sync_chirp)]
chirp2 = recording[position + len(sync_chirp) :position+len(sync_chirp)*2]
fft_chirp1 = np.fft.fft(chirp1)
fft_chirp2 = np.fft.fft(chirp2)

fft_sync_chirp = np.fft.fft(sync_chirp)

channel = fft_chirp2 / fft_sync_chirp
channel = channel[f0:f1]
# channel_old = channel
# channel_mag = scipy.signal.savgol_filter(np.absolute(channel),5,3)
# channel_angle = np.angle(channel)
# channel_zip = zip(channel_mag,channel_angle)
# channel = [ m*np.exp(1j*a) for m, a in channel_zip]
# channel = channel_old
# plt.plot(np.absolute(channel_old))
# # plt.plot(np.absolute(channel))
# plt.show()
channel = np.pad(channel,(f0,fs-f1))

impulse = np.fft.irfft(channel)


#perform least squares on the two chirps
x = np.linspace(f0,f1,f1-f0)
y = np.angle(fft_chirp2[f0:f1] * np.conj(fft_chirp1[f0:f1]))
sigma = np.std(y)
m, c = np.polyfit(x,y,1,w=1/(sigma**2))




# reverse channel effects
prefix_samples = fs
block_samples = fs

blocks = []
i=0
while True:
    print(i)
    group_length = prefix_samples + block_samples
    start = position_data + prefix_samples + group_length * i
    end = position_data + group_length + group_length * i
    data = recording[start:end]
    print(start,end,len(data))
    data_fft = np.fft.fft(data)
    data_fft = data_fft[f0:f1]
    data_fft = data_fft/(channel[f0:f1])

    #make cfo and sfo adjustment

    block_number = (i+1)*2

    for k in range(len(data_fft)):
        f =  f0 + k
        angle = np.exp(-block_number*1j * (m*f + c))
        data_fft[k] = data_fft[k] * angle

    print(data_fft)
    blocks.append(data_fft)
    i += 1
    if i == num_blocks:
        break





#decode signal
bytes_list, r_bits = d.blocks_to_bytes(blocks,4)

t_bits = e.random_binary(block_length*2*num_blocks)

#compare signals


for bl in range(len(blocks)):
    r = r_bits[bl*block_length*2:(bl+1)*2*block_length]
    t = t_bits[bl*block_length*2:(bl+1)*2*block_length]
    count = sum(1 for a,b in zip(r,t)if a != b) /(block_length*2) * 100
    errors = str(count)[:4] + "%"
    print("errors:",errors)
    view = 30
    print(" rec:",r[:view],"...",r[-view:])
    print("sent:",t[:view],"...",t[-view:])

for b in blocks:
    visualize.plot_fft(b,fs,f0,f1,title=f"{errors}")
    visualize.plot_constellation(b,title=f"{errors}")