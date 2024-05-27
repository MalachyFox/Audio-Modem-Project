import sounddevice as sd
import visualize
import numpy as np
import scipy
import playsound
import matplotlib.pyplot as plt
import decoder as d
import encoder as e
from py import ldpc

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
    freqs = np.linspace(N0,N0 + used_bins)
    angles = np.exp(-1j*coefs[0]+coefs[1])
    for k in range(len(data_fft)):
        f = N0 + k
        angle = np.exp(-1j*(coefs[0]*f + coefs[1]))
        data_fft_[k] *= angle
    # for k in range(len(data_fft)):
    #     f =  N0 + k
    #     angle = np.exp(-1j*(np.sum([a*f**b for a,b in zip(coefs,np.flip(list(range(len(coefs)))))]))) # ignore this unholy one liner to do polynomials
    #     data_fft[k] = data_fft[k] * angle
    return angles

def llhr(fft,channel_inv,sigma2_):

    yl = []
    for k in range(len(fft)):
        co =  np.real((1/channel_inv[k])*(np.conj(1/channel_inv[k]))/(sigma2_))
        #print(co)
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
    return fft

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
    for i in range(used_bins*2//ldpc_factor):
        chunk = binary[i*used_bins*2//ldpc_factor:(i+1)*used_bins*2//ldpc_factor]
        chunk = chunk[:2*used_bins_data//ldpc_factor]
        output.extend(chunk)
    return np.array(output)

def do_ldpc(data_fft,channel_inv,sigma2,pr=False):

    yl = llhr(data_fft,channel_inv,sigma2)
    app = []
    for i in range(ldpc_factor):
        app_temp, it = c.decode(yl[i*c.N:(i+1)*c.N],'sumprod2')
        if it > 199 and pr == True:
            print(" ! max it")
    app.extend(app_temp)
    binary = [ 1 if bitl > 0 else 0 for bitl in app]
    data_fft_ideal = binary_to_values(binary)

    return data_fft_ideal, it

### STANDARD ###
fs = 48000
block_length = 4096 
prefix_length = 512 
N0 = 100
###
recording_time = 14
chirp_factor = 16
c = ldpc.code('802.16','3/4',81)
ldpc_factor = 1
###
used_bins = (c.N//2)*ldpc_factor
chirp_length = block_length*chirp_factor
used_bins_data = (c.K//2)*ldpc_factor
N1 = N0+ used_bins
###
record = False
use_test_signal = False

def run(p):

    ### sync function ###
    sync_chirp = playsound.gen_chirp(N0,N0+used_bins,fs,chirp_length,block_length)
    sync = np.concatenate((sync_chirp[-prefix_length:],sync_chirp))


    ### start recording ###
    print("recording...",end="",flush=True)
    if record == True:
        input('press enter')
        recording = sd.rec(fs * recording_time,samplerate = fs,channels=1)
        sd.wait()
        recording = recording.flatten()
        playsound.save_signal(recording,fs,f'recordings/recording_{c.standard}_{c.N}_{c.K}_{N0}_{N1}.wav')
    else:
        if (use_test_signal):
            recording = playsound.load_signal(f'test_signals/test_signal_{c.standard}_{c.N}_{c.K}_{N0}_{N1}.wav')
        else:
            recording = playsound.load_signal(f'recordings/recording_{c.standard}_{c.N}_{c.K}_{N0}_{N1}.wav')
        recording = recording.flatten()
    print("done")




    ### find position ###
    print("synchronizing...",end="",flush=True)
    len_sync_chirp = len(sync_chirp)
    correlation = scipy.signal.correlate(recording[:fs*10], sync) # checks first 10s
    position = np.argmax(correlation) +1 # +1 moves slopes upwards CCW, seems to generally help?
    print("done")




    ### estimate channel ###
    print("estimating channel...",end="",flush=True)
    chirp = recording[position - len_sync_chirp :position]
    fft_chirp = get_fft_chirp(chirp)
    fft_sync_chirp = get_fft_chirp(sync_chirp)

    channel = fft_chirp/fft_sync_chirp
    channel = channel[N0:N0+used_bins]
    channel_inv = 1/channel
    channel_i = np.concatenate((np.ones(N0),channel,np.ones(block_length//2 + 1 - N0 - used_bins)))
    impulse = np.fft.irfft(channel_i)
    print("done")




    ### reverse channel effects ###
    print("channel correction...",end="\n",flush=True)
    blocks = []
    blocks_ideal= []
    order = 2
    sigma2 = 1
    block_index = 0
    group_length = prefix_length + block_length
    start = position + prefix_length
    end = position + group_length

    while True:
        print(f"\rblock: {block_index:04d}",end="")
        data = recording[start:end]
        if len(data) == 0:
            break
        #plt.plot(data)
        #plt.show()
        data_fft = np.fft.rfft(data)[:-1]
        data_fft = data_fft[N0:N0+used_bins]
        #print("\n",np.mean(np.absolute(data_fft)))
        data_fft *= channel_inv
        #print("\n",np.mean(np.absolute(data_fft)))
        #visualize.plot_fft(data_fft,fs)



        ## normalise first block and find sigma.
        
        if block_index == 0:
            power = np.mean(np.absolute(data_fft))
            data_fft /=power*np.sqrt(2)/2
            channel_inv /=power*np.sqrt(2)/2

        if block_index == 0:  # known block
            known_block_t = e.generate_known_block()[prefix_length:]
            data_fft_ideal = np.fft.rfft(known_block_t)[N0:N0+used_bins]
            power = np.mean(np.absolute(data_fft_ideal))
            data_fft_ideal /= power*np.sqrt(2)/2


            #print("\n",np.mean(np.absolute(data_fft_ideal)))#
            # visualize.plot_fft(data_fft_ideal,fs)
            # visualize.plot_fft(data_fft,fs)
            
            #sign = np.sign(np.real(np.mean(channel_inv))*np.imag(np.mean(channel_inv)))  # no idea why this has to be negative....
            
            complex_noise = data_fft_ideal - data_fft 
            sigma2 =  np.mean(np.imag(complex_noise)**2) + np.mean(np.real(complex_noise)**2)
            
            channel_inv *= (data_fft_ideal/data_fft)**(1/2)
            #data_fft *= (data_fft_ideal/data_fft)
            #print("\n",np.mean(np.absolute(data_fft)))


        ## do first ldpc
        print(sigma2)
        if block_index != 0:
            data_fft_ideal, it = do_ldpc(data_fft,channel_inv,sigma2)

            if it >199 and block_index > 5:  ##first can have too many errors, might be worth sending a warmup known block or a longer chirp?
                break
            
            channel_inv *= (data_fft_ideal/data_fft)**(1/2) #**(1/(1+a-(a/(b*block_index+1)))) # crazy function gives more weight at the start and less towards thte end, tapering to a constant 1/a with speed b a = 4, b = 0.05
            

            ## do linear shift
            inds = np.where(data_fft_ideal == 1 + 1j)[0]
            pilots = np.angle(data_fft[inds]) - np.pi/4
            freqs = inds + N0
            coefs_new = np.polyfit(freqs,pilots,order - 1)
            angles = apply_poly_to_fft(data_fft,coefs_new,N0)

            channel_inv *= angles
            data_fft *= angles

        
            complex_noise = data_fft_ideal - data_fft 
            sigma2 = np.mean(np.imag(complex_noise)**2) + np.mean(np.real(complex_noise)**2) # this also doesn't seem to make a difference
        
            data_fft_ideal, it = do_ldpc(data_fft,channel_inv,sigma2,pr=True)


        ## clean up
            blocks_ideal.append(data_fft_ideal)
            blocks.append(data_fft)

        start += group_length
        end += group_length
        block_index += 1
    num_blocks = block_index
    print("\ndone")



    ### decode signal ###
    print("decoding...",end="",flush=True)
    r_bits = blocks_to_binary(blocks_ideal)
    t_bits_information = e.random_binary(used_bins_data*2*(num_blocks -1))
    t_bits = e.encode_blocks(t_bits_information)
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



    ### compare signals ###
    t_bits = t_bits_information
    r_bits = decode(r_bits)
    
    total_errors = 0
    error_list = []
    for b in range(num_blocks):
        r = r_bits[b*used_bins_data*2:(b+1)*used_bins_data*2]
        t = t_bits[b*used_bins_data*2:(b+1)*used_bins_data*2]
        count = sum(1 for a,b in zip(r,t) if a != b)
        errors = count / (used_bins_data*2)
        error_list.append(errors)
        total_errors += errors
        if count !=0:
            print(f"block {b:04d} {count:02d} errors")
        # view = 20
        # print(" rec:",r[:view],"...",r[-view:])
        # print("sent:",t[:view],"...",t[-view:],"\n")

    total_errors = total_errors/num_blocks
    print(f"TOTAL ERRORS: {total_errors:.4%}")



    # # ### view plots ###
    # #visualize.big_plot([blocks[0],blocks[500],blocks[999]],fs,title="test",colours=np.concatenate((colours[0:used_bins],colours[500*used_bins:501*used_bins],colours[999*used_bins:1000*used_bins])))
    # #visualize.big_plot(blocks[:10],fs,title="test",colours=(colours[0:used_bins*10]))
    visualize.big_plot(blocks[:8],fs,title="test",colours=colours)
    visualize.plot_constellation(np.array(blocks).flatten(),colours=colours)

    plt.plot(correlation)
    plt.show()

    visualize.plot_channel(impulse)

    plt.plot(error_list)
    plt.ylim(0,20)
    plt.show()

    plt.plot(recording)
    plt.show()

    return total_errors


if __name__ == "__main__":
    # results = []
    # samples = list(range(50,250,10))
    # for var in samples:
    #     results.append(run(var))
    # plt.plot(samples,results)
    # plt.xlabel('clumping factor')
    # plt.ylabel(r'% errors')
    # plt.show()
    #print(scipy.optimize.dual_annealing(run,([1,50],[0,1])).x)

    run(1)


