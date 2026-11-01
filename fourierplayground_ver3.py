#v2 puts all the waves into one CosWave class to try make calculations happen faster using Numpy, because python classes can't be vectorised. The improvement seems negligible.
#also added file import and export
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog

import numexpr as ne

#file import from geeksforgeeks https://www.geeksforgeeks.org/python/create-an-import-file-button-with-tkinter/

selected_sample='testsound.mp3'

def import_file():
    global selected_sample
    global data
    global samplerate
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Audio Files", "*.mp3"), ("Audio files", "*.wav")])
    if file_path:
        # Process the selected file (you can replace this with your own logic)
        selected_sample=file_path
        data, samplerate = sf.read(selected_sample, dtype='float32')
        filename.configure(text=("Current File:", selected_sample))
        analyse(data, samplerate)
        slider.configure(from_=0, to=np.max(magnitudes), orient="horizontal", resolution=1e-7, command=update_selection, length=500)
        update_selection(0)

#https://www.plus2net.com/python/tkinter-filedialog-asksaveasfile.php
def save_data():
    file=filedialog.asksaveasfilename(initialfile='synthpreset.npz', defaultextension='.npz', filetypes=[('Numpy Archive', '*.npz')])
    if file:
        np.savez(file, frequencies=frequencies, magnitudes=magnitudes, angles=angles)

def save_sample():
    file=filedialog.asksaveasfilename(initialfile='fouriersample.wav', defaultextension='.wav', filetypes=[('Audio Files', '*.wav')])
    if file:
        sf.write(file, rebuilt_sound, samplerate)

#load audio sample
data, samplerate = sf.read(selected_sample, dtype='float32')

#interaction window
root=tk.Tk()
root.title("fourier playground")

og_graph=plt.figure(1, figsize=(7, 2))
og_graph_sub=og_graph.add_subplot()

new_graph=plt.figure(2, figsize=(7, 2))
graph_sub=new_graph.add_subplot()
plt.ion()

def analyse(data, samplerate):
    #combine stereo channels if there is any
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    #Compute the raw FFT
    global fft_output
    fft_output = np.fft.rfft(data)

    #Generate corresponding frequency bins
    global frequencies
    frequencies = np.fft.rfftfreq(len(data), d=1/samplerate)

    #Extract magnitudes
    global magnitudes
    magnitudes = np.abs(fft_output)

    #phase difference from cos shape
    global angles
    angles=np.angle(fft_output)

    #Plot the results
    global og_graph
    
    global og_graph_sub
    
    
    og_graph_sub.cla()
    og_graph_sub.plot(frequencies, magnitudes)
    og_graph_sub.set_title("Extracted Frequencies")
    og_graph_sub.set_xlabel("Frequency (Hz)")
    og_graph_sub.set_ylabel("Magnitude")
    og_graph_sub.grid()
    og_graph.canvas.draw()
   

    #output the actual usable frequencies
    global cutoff
    cutoff=0
    global important_mags
    global important_freqs
    global important_angs
    important_mags=magnitudes[magnitudes>cutoff]
    important_freqs= np.where(magnitudes>cutoff)[0]
    important_angs=angles[important_freqs]

    #plot the selected frequencies
    global new_graph
    
    global graph_sub
    
   
    graph_sub.cla()
    graph_sub.plot(important_freqs, important_mags)
    graph_sub.set_title("Selected Frequencies")
    graph_sub.set_xlabel("Frequency (Hz)")
    graph_sub.set_ylabel("Magnitude")
    graph_sub.grid()
    new_graph.canvas.draw()
    
    
   
analyse(data, samplerate)

#add to window

og_canvas = FigureCanvasTkAgg(og_graph, master=root)
og_plot_widget = og_canvas.get_tk_widget()
    

new_canvas = FigureCanvasTkAgg(new_graph, master=root)
new_plot_widget = new_canvas.get_tk_widget()

og_plot_widget.pack()
new_plot_widget.pack()


#rebuild the sound----------
rebuilt_sound=[]
def rebuild_sound():
    global rebuilt_sound
    global important_freqs
    global important_mags
    global fft_output
    #all indexes/frequencies that don't have enough magnitude
    mask = ~np.isin(np.arange(len(fft_output)), important_freqs)
    #set those frequencies to 0
    reduced_sample=np.where(mask, 0, fft_output)

    print("sample reduced")
    rebuilt_sound=np.fft.irfft(reduced_sample, n=len(data))
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
    global waves
    important_mags=magnitudes[magnitudes>float(cutoff)]
    important_freqs= np.where(magnitudes>float(cutoff))[0]
    important_angs=angles[important_freqs]
    print(len(important_mags), len(important_freqs), len(important_angs))
    graph_sub.plot(important_freqs, important_mags)
    graph_sub.set_title("Selected Frequencies")
    graph_sub.set_xlabel("Frequency (Hz)")
    graph_sub.set_ylabel("Magnitude")
    graph_sub.grid()
    rebuild_sound()
    #wave_list = vectorised_cos(important_freqs, important_mags, important_angs)
    waves=CosWave(important_freqs, important_mags, important_angs)
    

#initial recreation
rebuild_sound()

#################################################################################################################

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
        self.frequency = frequency[:,np.newaxis]
        self.amplitude = amplitude[:, np.newaxis]
        
        self.phase = phase[:,np.newaxis]

    def generate_block(self, frames, samplerate):
        global data
        #the phase is the angle of the wave. This calculates the increment for each timestep.
        phase_step = 2 * np.pi * self.frequency / len(data)
        
        self.frames=np.arange(frames)[np.newaxis,:]

        grid_phases=np.broadcast_to(self.phase, (len(self.phase), frames))

        #Generate a list of every phase in this block for the buffer

        frames2d=self.frames

        phases = ne.evaluate('grid_phases + frames2d * phase_step')
        
        #Step to the phase for the next block
        self.phase += frames * phase_step
        # Keep the phase bounded to avoid precision loss over time
        self.phase %= 2 * np.pi
        amplitude_grid=self.amplitude

        return ne.evaluate('amplitude_grid * cos(phases)')


# Define your list of sine waves (Frequency in Hz, Amplitude) -- replace with generate a list of the waves obtained from fourier transfrom
#vectorised_cos=np.vectorize(CosWave)
#wave_list = vectorised_cos(important_freqs, important_mags, important_angs)
waves=CosWave(important_freqs, important_mags, important_angs)

def audio_callback(outdata, frames, time, status, blocksize=2496):
    if status:
        print(status)
    global important_mags
    # Start with a silent block of zeros
    #mixed_buffer = np.zeros(frames)
    
    # Mathematically add each wave's array into the master buffer.
    #the frames are every sample step for that block
    '''for wave in wave_list:
        mixed_buffer += wave.generate_block(frames, samplerate)'''
    wave_data=waves.generate_block(frames, samplerate)
    mixed_buffer=ne.evaluate('sum(wave_data, axis=0)')
    

    # Write the mixed mono signal into the stream's output channel(s)
    divisor=ne.evaluate('sum(important_mags)')
    outdata[:, 0] = ne.evaluate('mixed_buffer/divisor')  #(len(important_freqs)*np.max(magnitudes))

print("synthesis initialised")

#Initialize the stream without starting it immediatelys
stream = sd.OutputStream(channels=1, callback=audio_callback, samplerate=samplerate)

# Track the current state of our stream
is_playing = False
#use stream.start()and stream.stop() to control playback.
synthesize=tk.Button(root, text="synthesize", command=lambda: stream.start())
stop_synthesis=tk.Button(root, text="stop synthesis", command=lambda: stream.stop())
synthesize.pack()
stop_synthesis.pack()

#import a selected sample
import_button = tk.Button(root, text="Import File", command=import_file)
import_button.pack()

filename=tk.Label(root, text=("Current File:", selected_sample))
filename.pack()

#save the data of the filtered wave set
save_button = tk.Button(root, text="Save synth preset", command=save_data)
save_button.pack()

#save the individual filtered audio sample
export_button = tk.Button(root, text="Save audio sample", command=save_sample)
export_button.pack()


#####################################################################


root.mainloop()
