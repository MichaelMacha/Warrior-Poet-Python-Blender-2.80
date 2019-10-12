from skimage import util
import numpy as np
from scipy import fftpack
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path
import os

rate, audio = wavfile.read(os.path.join(Path.home(), 'Downloads/186942__lemoncreme__piano-melody.wav'))
audio = np.mean(audio, axis=1)
N = audio.shape[0]
L = N / rate
M = 1024
slices = util.view_as_windows(audio, window_shape=(M, ), step=100)
win = np.hanning(M + 1)[:-1]
slices = (slices * win).T
spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
spectrum = np.abs(np.fft.fft(slices, axis=0)[:M // 2 + 1:-1])
S = np.abs(spectrum)
S = 20 * np.log10(S / np.max(S))

f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.imshow(S, origin='lower', cmap='viridis',
          extent=(0, L, 0, rate / 2 / 1000))
ax.axis('tight')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');
plt.show()

