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

seconds = 8
fs = 44100
gain = 2
f0 = 1000
block_length = 16384
f1 = f0 + block_length


#generate double sync function
sync = playsound.gen_chirp(f0,f1,fs,1)
sync = playsound.double_signal(sync)


#start recording
input = input('press space')
recording = sd.rec(fs * seconds,samplerate = fs,channels=1)
sd.wait()
recording = recording.flatten()


# find position
correlation = scipy.signal.correlate(recording, sync)
peak_correlation = np.max(correlation)
position = np.argmax(correlation) - len(sync)

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
chirp1 = recording[position : position + len(sync)//2]
chirp2 = recording[position + len(sync)//2 :position+len(sync)]
fft_chirp1 = np.fft.fft(chirp1)
fft_chirp2 = np.fft.fft(chirp2)

sync2 = sync[len(sync)//2:]
fft_sync2 = np.fft.fft(sync2)

channel = fft_chirp2 / fft_sync2
channel = channel[f0:f1]
channel = np.pad(channel,(f0,fs-f1))

impulse = np.fft.irfft(channel)


# reverse channel effects
data = recording[position+len(sync)+fs:position+len(sync)+fs*2]
data_fft = np.fft.fft(data)
data_fft = data_fft[f0:f1]
data_fft = data_fft/(channel[f0:f1])


#perform least squares on the two chirps
x = np.linspace(f0,f1,f1-f0)
y = np.angle(fft_chirp2[f0:f1] * np.conj(fft_chirp1[f0:f1]))
A = np.vstack([x,np.ones(len(x))]).T
m, c = np.linalg.lstsq(A,y,rcond=None)[0]

#make cfo and sfo adjustment

block_number = 1

for i in range(len(data_fft)):
    f =  f0 + i
    angle = np.exp(-block_number*2*1j *(f*m+c))
    data_fft[i] = data_fft[i] * angle


#decode signal
bytes_list, r_bits = d.blocks_to_bytes([data_fft],4)

t_bits = e.random_binary(block_length*2)

#compare signals
print("   received bits:",r_bits[:30])
print("transmitted bits:",t_bits[:30])
count = sum(1 for a,b in zip(r_bits,t_bits) if a != b) /(block_length*2) * 100
errors = str(count)[:4] + "%"
print("errors:",errors)

visualize.plot_fft(data_fft,fs,f0,f1,title=f"fft_{f0}_{f1}_{errors}")
visualize.plot_constellation(data_fft,title=f"fft_{f0}_{f1}_{errors}")