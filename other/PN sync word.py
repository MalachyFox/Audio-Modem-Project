import numpy as np
from scipy.io import wavfile
import csv

def generate_sync_word_time_domain(pn_length):

    pn = np.random.randint(2, size=pn_length) * 2 - 1

    sync_word_time_domain = np.fft.ifft(pn)
    sync_word_time_domain = np.tile(sync_word_time_domain, 2) 

    return sync_word_time_domain

pn_length = 1000  

# Generate synchronization word in the time domain
sync_word_time_domain = generate_sync_word_time_domain(pn_length)

# Scale the sync word to be in the range of [-1, 1]
sync_word_time_domain = sync_word_time_domain / np.max(np.abs(sync_word_time_domain))

#sync_word_time_domain = np.random.random(pn_length*2) * 2 - 1

#Convert to 16-bit integer format
sync_word_time_domain = np.int16(sync_word_time_domain * 32767)

with open("/Users/lit./Desktop/gf3/gf3 2/syncpn.csv", "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(sync_word_time_domain)
    
# Save sync word as WAV file
wavfile.write('/Users/lit./Desktop/gf3/gf3 2/syncpn.wav', 44100, sync_word_time_domain)

import matplotlib.pyplot as plt
plt.plot(sync_word_time_domain)
plt.title('Synchronization Word')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

plt.plot(np.absolute(np.fft.fft(sync_word_time_domain)))
plt.grid(True)
plt.show()