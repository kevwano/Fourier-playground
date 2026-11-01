#v1 creates a CosWave class for each frequency and then adds them together during the audio streaming process
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import soundfile as sf
import sounddevice as sd
import tkinter as tk

#load audio sample
data, samplerate = sf.read('testsound.mp3')

#combine stereo channels, if there is
if len(data.shape) > 1:
    data = np.mean(data, axis=1)

#Compute the raw FFT
fft_output = np.fft.rfft(data)

#Generate corresponding frequency bins
frequencies = np.fft.rfftfreq(len(data), d=1/samplerate)

#Extract magnitudes
magnitudes = np.abs(fft_output)

#phase difference from cos shape
angles=np.angle(fft_output)

#Plot the results
og_graph=plt.figure(1, figsize=(7, 2))
plt.ion()
plt.plot(frequencies, magnitudes)
plt.title("Extracted Frequencies")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.draw()

#interaction window
root=tk.Tk()
root.title("fourier playground")
#root.geometry("600x300")
og_canvas = FigureCanvasTkAgg(og_graph, master=root)
og_plot_widget = og_canvas.get_tk_widget()
og_plot_widget.pack()

#output the actual usable frequencies
cutoff=0
important_mags=magnitudes[magnitudes>cutoff]
important_freqs= np.where(magnitudes>cutoff)[0]
important_angs=angles[important_freqs]

#plot the selected frequencies
new_graph=plt.figure(2, figsize=(7, 2))
graph_sub=new_graph.add_subplot()
plt.ion()
graph_sub.plot(important_freqs, important_mags)
graph_sub.set_title("Selected Frequencies")
graph_sub.set_xlabel("Frequency (Hz)")
graph_sub.set_ylabel("Magnitude")
graph_sub.grid()
new_graph.canvas.draw()
plt.close()

#add to window
new_canvas = FigureCanvasTkAgg(new_graph, master=root)
new_plot_widget = new_canvas.get_tk_widget()
new_plot_widget.pack()

#rebuild the sound----------
rebuilt_sound=[]
def rebuild_sound():
    global rebuilt_sound
    global important_freqs
    global important_mags
    #all indexes/frequencies that don't have enough magnitude
    mask = ~np.isin(np.arange(len(fft_output)), important_freqs)
    #set those frequencies to 0
    reduced_sample=np.where(mask, 0, fft_output)

    print("sample reduced")
    rebuilt_sound=np.fft.irfft(reduced_sample, n=samplerate)
    #create a playable output
    sf.write("output.wav", rebuilt_sound, samplerate)
    print(rebuilt_sound)
    print("output done")

#update when slider is moved
def update_selection(cutoff):
    stream.stop()
    graph_sub.cla()
    global important_mags
    global important_freqs
    global wave_list
    important_mags=magnitudes[magnitudes>float(cutoff)]
    important_freqs= np.where(magnitudes>float(cutoff))[0]
    important_angs=angles[important_freqs]
    graph_sub.plot(important_freqs, important_mags)
    graph_sub.set_title("Selected Frequencies")
    graph_sub.set_xlabel("Frequency (Hz)")
    graph_sub.set_ylabel("Magnitude")
    graph_sub.grid()
    rebuild_sound()
    wave_list = vectorised_cos(important_freqs, important_mags, important_angs)

#initial recreation
rebuild_sound()

#playback buttons
def play_sound(sound):
    sd.play(sound, samplerate)

playogbutton=tk.Button(root, text="play original sound", command=lambda: play_sound(data))
playogbutton.pack()

playnewbutton=tk.Button(root, text="play new sound", command=lambda: play_sound(rebuilt_sound))
playnewbutton.pack()

#adjust cutoff threshold
slider=tk.Scale(root, from_=0, to=np.max(magnitudes), orient="horizontal", resolution=1e-7, command=update_selection, length=500)
slider.pack()

#Synthesise using the found frequencies ##############################

class CosWave:
    def __init__(self, frequency, amplitude, phase=0):
        self.frequency = frequency
        self.amplitude = amplitude
        
        self.phase = phase

    def generate_block(self, frames, samplerate):
        #the phase is the angle of the wave. This calculates the increment for each timestep.
        phase_step = 2 * np.pi * self.frequency / samplerate
        
        #Generate a list of every phase in this block for the buffer
        phases = self.phase + np.arange(frames) * phase_step
        
        #Step to the phase for the next block
        self.phase += frames * phase_step
        # Keep the phase bounded to avoid precision loss over time
        self.phase %= 2 * np.pi
        
        return self.amplitude * np.cos(phases)


#generate a list of the waves obtained from fourier transfrom
vectorised_cos=np.vectorize(CosWave)
wave_list = vectorised_cos(important_freqs, important_mags, important_angs)


def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
        
    # Start with a silent block of zeros
    mixed_buffer = np.zeros(frames)
    
    # Mathematically add each wave's array into the master buffer.
    #the frames are every sample step for that block
    for wave in wave_list:
        mixed_buffer += wave.generate_block(frames, samplerate)
        
    # Write the mixed mono signal into the stream's output channel(s)
    outdata[:, 0] = mixed_buffer/(len(important_freqs)*np.max(magnitudes))

print("synthesis initialised")

#Initialize the stream without starting it immediately
stream = sd.OutputStream(channels=1, callback=audio_callback, samplerate=samplerate)

# Track the current state of our stream
is_playing = False
#use stream.start()and stream.stop() to control playback.
synthesize=tk.Button(root, text="synthesize", command=lambda: stream.start())
stop_synthesis=tk.Button(root, text="stop synthesis", command=lambda: stream.stop())
synthesize.pack()
stop_synthesis.pack()

#####################################################################


root.mainloop()
