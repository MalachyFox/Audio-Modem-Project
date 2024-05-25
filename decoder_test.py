import sounddevice as sd
import visualize
import numpy as np
import scipy
import playsound
import matplotlib.pyplot as plt
import decoder as d
import encoder as e

def get_fft_chirp(chirp,overlap = False):
    fft_chirp = np.zeros(block_length//2,dtype=np.complex_)
    if overlap == True:
        for i in range(chirp_factor* 2 - 1):
            partial_chirp = chirp[i*block_length//2:(i+2)*block_length//2]
            partial_chirp *= scipy.signal.windows.hamming(block_length)
            fft_partial_chirp = np.fft.rfft(partial_chirp)[:-1]
            fft_chirp += fft_partial_chirp
    else:
        for i in range(chirp_factor):
            partial_chirp = chirp[i*block_length:(i+1)*block_length]
            fft_partial_chirp = np.fft.rfft(partial_chirp)[:-1]
            fft_chirp += fft_partial_chirp

    return fft_chirp/np.max(fft_chirp)

def remove_tracking(binary):
    length = used_bins*bpv//tracking_length
    output = ""
    list_of_lists = [binary[(i*length)+bpv:(i+1)*length] for i in range(num_blocks*tracking_length)]
    for list in list_of_lists:
        for bit in list:
            output += bit
    return output

def apply_poly_to_fft(data_fft,coefs,N0):
    data_fft_ = data_fft
    for k in range(len(data_fft)):
        f = N0 + k
        angle = np.exp(-1j*(coefs[0]*f + coefs[1]))
        data_fft_[k] *= angle
    # for k in range(len(data_fft)):
    #     f =  N0 + k
    #     angle = np.exp(-1j*(np.sum([a*f**b for a,b in zip(coefs,np.flip(list(range(len(coefs)))))]))) # ignore this unholy one liner to do polynomials
    #     data_fft[k] = data_fft[k] * angle
    return data_fft_

### STANDARD ###
fs = 48000
block_length = 4096 
bpv = 2 # bits per value 
prefix_length = 512 
N0 = 85
N1 = 850
###
recording_time = 100
chirp_factor = 16
tracking_length = 15
num_blocks = 1000
###
used_bins = N1 - N0
chirp_length = block_length * chirp_factor
used_bins_data = used_bins - tracking_length
###
record = False
use_test_signal = False


def run():
    


    ### sync function ###
    sync_chirp = playsound.gen_chirp(N0,N1,fs,chirp_length,block_length)
    sync = np.concatenate((sync_chirp[-prefix_length:],sync_chirp))



    ### start recording ###
    print("recording...",end="",flush=True)
    if record == True:
        input('press enter')
        recording = sd.rec(fs * recording_time,samplerate = fs,channels=1)
        sd.wait()
        recording = recording.flatten()
        playsound.save_signal(recording,fs,f'recordings/recording_{chirp_factor}c_{tracking_length}t_{num_blocks}b.csv')
    else:
        if (use_test_signal):
            recording = playsound.load_signal(f'test_signals/test_signal_{chirp_factor}c_{tracking_length}t_{num_blocks}b.wav')
            
        else:
            recording = playsound.load_signal(f'recordings/recording_{chirp_factor}c_{tracking_length}t_{num_blocks}b.wav') #   #(f'recordings/recording_{f0}_{f1}_{num_blocks}b.csv') #
            #playsound.save_signal(recording,fs,f'recordings/recording_{chirp_factor}c_{tracking_length}t_{num_blocks}b.wav')

        recording = recording.flatten()
    print("done")


    ### find position ###
    print("synchronizing...",end="",flush=True)
    len_sync_chirp = len(sync_chirp)
    correlation = scipy.signal.correlate(recording, sync)
    position = np.argmax(correlation) + 1 # +1 moves slopes upwards CCW
    print("done")


    ### estimate channel ###
    print("estimating channel...",end="",flush=True)
    chirp = recording[position - len_sync_chirp :position]
    #chirp *= scipy.signal.windows.hamming(block_length)

    fft_chirp = get_fft_chirp(chirp)
    fft_sync_chirp = get_fft_chirp(sync_chirp)
    channel = fft_chirp/fft_sync_chirp

    channel = channel[N0:N1]
    channel_i = np.concatenate((np.ones(N0),channel,np.ones(block_length//2- N1)))
    impulse = np.fft.irfft(channel_i)
    print("done")


    ### reverse channel effects ###
    print("channel correction...",end="",flush=True)
    blocks = []
    block_index = 0
    order = 2
    coefs = np.zeros(order)

    group_length = prefix_length + block_length
    spacing = used_bins//(tracking_length)
    while True:

        start = position + prefix_length + group_length * block_index
        end = position + group_length + group_length * block_index
        data = recording[start:end]

        data_fft = np.fft.rfft(data)
        data_fft = data_fft[N0:N1]
        data_fft = data_fft/(channel)

        data_fft = apply_poly_to_fft(data_fft,coefs,N0)

        pilot_indices = np.array([i*spacing for i in range(tracking_length)])
        pilots = np.array([np.angle(data_fft[i]) for i in pilot_indices]) - np.pi/4 # or array of pilot locations
        freqs = pilot_indices + N0
        coefs_new = np.polyfit(freqs,pilots,order - 1)

        data_fft = apply_poly_to_fft(data_fft,coefs_new,N0)
        
        coefs += coefs_new


        mean = []                       ## this random little section looks for the mean of the top right values,
        for value in data_fft:          ## and adjusts the whole plot so theyre in the centre, gives -0.08% error
            value = np.angle(value)     ## improvement so i guess its staying haha (-0.02% with above algorithm as well)
            if value > 0 and value < np.pi/2:
                mean.append(value)
        mean = np.sum(mean)/len(mean) - np.pi/4
        data_fft = data_fft*np.exp(-1j*mean)


        estimate = np.ones(used_bins,dtype=np.complex_)
        for k in range(used_bins):
            value = np.angle(data_fft[k])
            if value > 0  and value < np.pi/2 :
                new_value = np.pi/4
            elif value > np.pi/2  and value < np.pi :
                new_value = np.pi/4 + np.pi/2
            elif value < 0  and value > -np.pi/2 :
                new_value = -np.pi/4
            elif value < -np.pi/2  and value > -np.pi :
                new_value = -np.pi/4 - np.pi/2

            estimate[k] = np.exp(-1j*(new_value - value)/128)
        

        channel *= estimate
                

  

        blocks.append(data_fft)

        block_index += 1
        if block_index == num_blocks:
            break
    print("done")



    ### decode signal ###
    print("decoding...",end="",flush=True)
    bytes_list, r_bits = d.blocks_to_bytes(blocks)
    r_bits = remove_tracking(r_bits)
    t_bits = e.random_binary(used_bins_data*bpv*num_blocks)
    t_bits = e.add_tracking(t_bits)
    t_bits = e.correct_binary_length(t_bits)
    print("done")



    ### add colours ###         # not scalable for M-ary yet
    colours = []
    for i in range(len(t_bits)//bpv):
        bit = t_bits[i*bpv:(i+1)*bpv]
        if bit == "00":
            colours.append("r")
        elif bit == "01":
            colours.append("y")
        elif bit == "11":
            colours.append("g")
        elif bit == "10":
            colours.append("b")





    ### compare signals ###
    t_bits = remove_tracking(t_bits)
    total_errors = 0
    error_list = []
    for b in range(len(blocks)):
        r = r_bits[b*used_bins_data*bpv:(b+1)*used_bins_data*bpv]
        t = t_bits[b*used_bins_data*bpv:(b+1)*used_bins_data*bpv]
        count = sum(1 for a,b in zip(r,t)if a != b) /(used_bins_data*bpv) * 100
        error_list.append(count)
        total_errors += count
        errors = str(count)[:4] + "%"
        #print(f"block {b}, {errors} errors")
        # view = 20
        # print(" rec:",r[:view],"...",r[-view:])
        # print("sent:",t[:view],"...",t[-view:],"\n")

    total_errors = total_errors/num_blocks
    print(f"TOTAL ERRORS: {(str(total_errors))[:4]}%")



    ### view plots ###
    #visualize.big_plot([blocks[0],blocks[500],blocks[999]],fs,title="test",colours=np.concatenate((colours[0:used_bins],colours[500*used_bins:501*used_bins],colours[999*used_bins:1000*used_bins])))
    visualize.big_plot(blocks[:10],fs,title="test",colours=(colours[0:used_bins*10]))

    full_range = []
    for b in blocks:
        full_range = np.concatenate((full_range,b))

    #visualize.plot_fft(full_range,fs,colours)
    visualize.plot_constellation(full_range,colours=colours)

    plt.plot(correlation)
    plt.show()

    visualize.plot_channel(impulse)

    plt.plot(error_list)
    plt.ylim(0,20)
    plt.show()

    return total_errors


if __name__ == "__main__":
    run()


