import sounddevice as sd
import visualize
import numpy as np
import scipy.signal
import playsound
import matplotlib.pyplot as plt
import encoder as e
from py import ldpc
import time
import librosa

def get_fft_chirp(chirp,overlap = False):
    fft_chirp = np.zeros(block_length,dtype=np.complex_)
    if overlap == True:
        for i in range(chirp_factor* 2 - 1):
            partial_chirp = chirp[i*block_length//2:(i+2)*block_length//2]
            partial_chirp *= scipy.signal.windows.hamming(block_length)
            fft_partial_chirp = np.fft.fft(partial_chirp)
            fft_chirp += fft_partial_chirp
    else:
        for i in range(chirp_factor):
            partial_chirp = chirp[i*block_length:(i+1)*block_length]
            fft_partial_chirp = np.fft.fft(partial_chirp)
            fft_chirp += fft_partial_chirp

    return fft_chirp /chirp_factor

def apply_poly_to_fft(data_fft,coefs,N0):
    data_fft_ = data_fft
    freqs = np.linspace(N0,N0 + used_bins,used_bins,endpoint = False)
    angles = np.exp(-1j*(coefs[0]*freqs+coefs[1]))
    # for k in range(len(data_fft)):
    #     f = N0 + k
    #     angle = np.exp(-1j*(coefs[0]*f + coefs[1]))
    #     data_fft_[k] *= angle
    # for k in range(len(data_fft)):
    #     f =  N0 + k
    #     angle = np.exp(-1j*(np.sum([a*f**b for a,b in zip(coefs,np.flip(list(range(len(coefs)))))]))) # ignore this unholy one liner to do polynomials
    #     data_fft[k] = data_fft[k] * angle
    return angles

def llhr(fft,channel_inv,sigma2_,power):
    yl = []
    for k in range(len(fft)):
        co =  np.real((1/channel_inv[k])*(np.conj(1/channel_inv[k]))/(sigma2_))
        l1 = np.sqrt(2)*np.imag(fft[k])*co
        l2 = np.sqrt(2)*np.real(fft[k])*co
        yl.append(l1)
        yl.append(l2)
    #print(yl[:4])
    return np.array(yl)

def binary_to_values(binary):
    fft = np.zeros(used_bins,dtype=np.complex_)
    for k in range(used_bins):
        value = binary[k*2:(k+1)*2]
        point = 0 + 0j
        if value[0] > 0:
            point+= 1j
        else:
            point -= 1j
        if value[1] > 0:
            point += 1
        else:
            point -= 1
        fft[k] = point 
    return fft/np.sqrt(2)

def blocks_to_binary(blocks_):
    binary_big = []
    for block in blocks_:
        binary = []
        for value in block:
            output = [1,1]
            if np.imag(value) >0:
                output[0] = 0
            if np.real(value) > 0:
                output[1] = 0
            binary.extend(output)
        binary_big.extend(binary)
    return np.array(binary_big)

def decode(binary):
    output = []
    for i in range(len(binary)//(used_bins*2)):
        chunk = binary[i*used_bins*2//ldpc_factor:(i+1)*used_bins*2//ldpc_factor]
        chunk = chunk[:2*used_bins_data//ldpc_factor]
        output.extend(chunk)
    return np.array(output)

def do_ldpc(data_fft,channel_inv,sigma2,power,pr=False):

    yl = llhr(data_fft,channel_inv,sigma2,power)
    app = []
    for i in range(ldpc_factor):
        app_temp, it = c.decode(yl[i*c.N:(i+1)*c.N],'sumprod2')
        if it > 199 and pr == True:
            print(" ! max it")
    app.extend(app_temp)
    binary = [ 1 if bitl > 0 else 0 for bitl in app]  #??????
    data_fft_ideal = binary_to_values(binary)

    return data_fft_ideal, it

### STANDARD ###
fs = 48000
block_length = 4096 
prefix_length = 1024 
B0 = 85
###
recording_time = 60*7 + 36 + 10
chirp_factor = 16
c = ldpc.code('802.16','1/2',54)
ldpc_factor = 1
###
used_bins = (c.N//2)*ldpc_factor
chirp_length = block_length*chirp_factor
used_bins_data = (c.K//2)*ldpc_factor
B1 = B0 + used_bins
###
record = False
use_test_signal = True
filename_="moomoo.bmp.wav"


def run(snr,factor,do_angle,do_channel):

    ### sync function ###
    sync_chirp = playsound.gen_chirp(B0,B1,fs,chirp_length,block_length)
    sync = np.concatenate((sync_chirp[-prefix_length:],sync_chirp,sync_chirp[:prefix_length]))*0.1

    ### start recording ###
    print("recording...",end="",flush=True)
    if record == True:
        input('press enter')
        recording = sd.rec(fs * recording_time,samplerate = fs,channels=1)
        sd.wait()
        recording = recording.flatten()
        playsound.save_signal(recording,fs,f'recordings/'+ filename_)
    else:
        if (use_test_signal):
            recording = playsound.load_signal(f'test_signals/' + filename_)
        else:
            recording = playsound.load_signal(f'recordings/' + filename_)
        recording = recording.flatten()
    # print()
    # for t in range(recording_time):
    #     print(f'{t}s',end='\r')
    #     time.sleep(1)

    signalP = np.mean(np.absolute(recording[200000:300000])**2)
    #print(signalP)
    SNR = 1.3
    N_0 = signalP/SNR
    noise_sigma = np.sqrt(N_0/2)

    recording = librosa.resample(recording,orig_sr=48000,target_sr = 48000+2)

    recording += np.random.normal(0,noise_sigma,len(recording))


    print("done")
    # plt.plot(recording)
    # plt.show()

    ### find position ###
    print("synchronizing...",end="",flush=True)
    len_sync_chirp = len(sync_chirp)
    correlation = scipy.signal.correlate(recording[:fs*20], sync) # checks first 10s
    position = np.argmax(correlation) + 1 # +1 moves slopes upwards CCW, seems to generally help?
    print("done")

    ### estimate channel ###
    print("estimating channel...",end="",flush=True)
    chirp = recording[position - len_sync_chirp -prefix_length : position-prefix_length]
    fft_chirp = get_fft_chirp(chirp)
    fft_sync_chirp = get_fft_chirp(sync_chirp)
  
    # visualize.plot_fft(fft_chirp,fs)
    # visualize.plot_fft(fft_sync_chirp,fs)

    channel = fft_chirp[B0:B1]/fft_sync_chirp[B0:B1]

    channel_inv = 1/channel

    channel_i = np.concatenate((np.ones(B0),channel,np.ones(block_length//2 + 1 - B0 - used_bins)))
    impulse = np.fft.irfft(channel_i)
    print("done")


    ### reverse channel effects ###
    print("channel correction...",end="\n",flush=True)
    
    order = 2
    sigma2 = 1
    
    group_length = prefix_length + block_length
    start = position + prefix_length
    end = position + group_length

    known_block_t, known_block_fft = e.generate_known_block()
    known_block_t = known_block_t[prefix_length:]
    known_block_fft = known_block_fft[B0:B1]
    known_block_fft_norm = known_block_fft/ np.sqrt(np.mean(np.absolute(known_block_fft))**2)
    sigmas = []
    blocks = []
    blocks_ideal = []
    block_index = 0
    angles = np.ones(used_bins)
    its = np.array([])
    power = 0
    fail_after = 5

    while True:
        print(f"\rblock: {block_index:04d}",end="")
        data = recording[start:end]
        if len(data) == 0:
            break
        data_fft = np.fft.rfft(data)
        #power2 = np.mean(np.absolute(data_fft)**2)
        data_fft = data_fft[B0:B1]
        power2 = np.mean(np.absolute(data_fft)**2)
        #print('\n',power2)
        data_fft *= channel_inv
        
        if block_index == 0:

            power = np.sqrt(np.mean(np.absolute(data_fft))**2)
            data_fft /= power
            channel_inv /= power

            data_fft_ideal = known_block_fft
            complex_noise =  data_fft/channel_inv - known_block_fft/channel_inv
            
            sigma2 =  0.5*np.mean(np.absolute(complex_noise)**2) # 0.5 represents 1/Amplitude**2 amp is sqrt2
            channel_inv_adj = (known_block_fft/data_fft)**(1)
            channel_inv *=channel_inv_adj
        
        if block_index != 0:
            #print('\n',power2/(2*2*sigma2))
            data_fft /= (known_block_fft_norm/np.exp(1j*np.pi/4))

            data_fft_ideal, it = do_ldpc(data_fft,channel_inv,sigma2,power)

            ## do linear shift
            inds = np.where(np.angle(data_fft_ideal) == np.pi/4)[0]
            pilots = np.angle(data_fft[inds]) - np.pi/4
            freqs = inds + B0
            coefs_new = np.polyfit(freqs,pilots,order - 1)
            angles = apply_poly_to_fft(data_fft,coefs_new,B0)
            if do_angle:
                channel_inv *= angles
                data_fft *= angles

            complex_noise = data_fft_ideal/channel_inv - data_fft/channel_inv
            complex_noise_average = np.mean(np.absolute(complex_noise)**2)
            sigma2 = 0.5*complex_noise_average
            
            data_fft_ideal, it = do_ldpc(data_fft,channel_inv,sigma2,power,pr=True)
            
            if do_channel:
                channel_inv *= (data_fft_ideal/data_fft)**(1/factor) # soft update / clustering

            its = np.append(its,it)
            # if np.sum(its[-fail_after:]) >= 50*fail_after:
            #     break


            ## clean up
            blocks_ideal.append(data_fft_ideal)
            blocks.append(data_fft)

        start += group_length
        end += group_length
        block_index += 1
        if block_index == 209:
            break
   
    num_blocks = block_index - fail_after
    print("\ndone")
    print("iterations:",int(np.sum(its)))


    ### decode signal ###
    print("decoding...",end="",flush=True)
    r_bits = blocks_to_binary(blocks_ideal)
    filename = 'moomoo.bmp'
    t_bits = e.load_file(filename)
    t_bits = e.add_header(t_bits,filename)
    t_bits = e.correct_binary_length(t_bits)
    t_bits = e.encode_blocks(t_bits)
    print("done")

    ### add colours ###    
    colours = []
    for i in range(len(r_bits)//2):
        bit = list(r_bits[i*2:(i+1)*2])  # r_bits for guessed colours, t_bits for known colours
        if (bit == [0,0]):
            colours.append("r")
        elif (bit == [0,1]):
            colours.append("y")
        elif (bit == [1,1]):
            colours.append("g")
        elif (bit == [1,0]):
            colours.append("b")

    # # ### compare signals ###
    t_bits = decode(t_bits)
    r_bits = decode(r_bits)
    
    total_errors = 0
    error_list = []
    for b in range(num_blocks - 1):
        r = r_bits[b*used_bins_data*2:(b+1)*used_bins_data*2]
        t = t_bits[b*used_bins_data*2:(b+1)*used_bins_data*2]
        count = sum(1 for a,b in zip(r,t) if a != b)
        errors = count / (used_bins_data*2)
        error_list.append(errors)
        total_errors += errors
        #if count !=0:
            #print(f"block {b:04d} {count:02d} errors")
        # view = 20
        # print(" rec:",r[:view],"...",r[-view:])
        # print("sent:",t[:view],"...",t[-view:],"\n")

    total_errors = total_errors/num_blocks
    print(f"TOTAL ERRORS: {total_errors:.4%}")


    # ### HAORAN SECTION ###
    # bytes_list = []
    # for i in range(len(r_bits//8)):
    #     byte = r_bits[i*8:(i+1)*8]
    #     try:
    #         byte_int = 0
    #         for ii in range(8):
    #             byte_int += (2**(7-ii))*byte[ii]
    #         bytes_list.append(byte_int)
    #     except:
    #         pass
    
    # from PIL import Image
    # im = Image.frombuffer('RGB', (54,54), np.array(bytes_list,dtype='int8'), 'raw', 'RGB', 0, 1)
    # im.save("haroan.png")
                
    
    filename, size, data = e.handle_header(r_bits)
    print(filename,size,data[:20])
    try:
        with open("./received_files/" + filename,"wb") as output_file:
            output_file.write(data)
    except:
        print("FAILED TO SAVE")

    #see = [int(a) for a in np.linspace(0,num_blocks - 1 - fail_after,5)]
    #see = [1,2,3,4]
    # plt.style.use('ggplot')
    # visualize.big_plot([blocks[i] for i in see],fs,title="test",colours=np.array([colours[i*used_bins:(i+1)*used_bins] for i in see]).flatten())
    # visualize.plot_constellation(np.array(blocks).flatten(),colours=colours)

    
    #plt.plot(correlation)
    #plt.show()

    #visualize.plot_channel(impulse)

    # plt.plot(error_list)
    # plt.ylim(0,20)
    # plt.show()

    #plt.plot(recording)
    #plt.show()

    return total_errors


if __name__ == "__main__":
    run(2,10,True,True)
    plt.style.use('ggplot')
    plt.rcParams["font.family"] = "serif"

    # results = []
    # samples = np.logspace(0,5,30,endpoint=True)
    # for var in samples:
    #     print(var)
    #     results.append(run(1,var,True,True))
    # print(samples)
    # print(results)
    # plt.loglog()
    # plt.plot(samples,results,linestyle='solid',lw=3)
    # #plt.ylim(0,1)
    # # plt.legend(['Linear phase compensation and channel estimate update',
    # #             'Channel estimate update',
    # #             'Linear phase compensation',
    # #             'No strategy'],fontsize=15)
    
    # plt.xlabel('Channel update factor',fontsize=15)
    # plt.ylabel(r'Bit error rate',fontsize=15)
    # plt.savefig('test2')
    # plt.show()
    #print(scipy.optimize.dual_annealing(run,([1,50],[0,1])).x)






