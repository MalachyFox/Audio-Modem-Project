import playsound as ps

fs = 44100

signal = ps.gen_chirp( 3500,4500,fs,1)
signal = ps.double_signal(signal)
ps.play_signal(signal,fs)