# author: Martin Fouilleul

import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
import pyglim as gl

# simple stft to produce our test spectrogram


def test_stft(x, window, hop_size):

    n_fft = len(window)
    spectrum = np.array([np.fft.rfft(window*x[i:i+n_fft])
                         for i in range(0, len(x)-n_fft, hop_size)])
    return spectrum

# Build test signal


fe = 44100.
length_seconds = 0.5
n_samples = int(fe * length_seconds)

f0 = 1000. / fe

x_ref = 0.5 * np.sin(2 * np.pi * f0 * np.arange(0, n_samples))
x_ref += 0.1 * np.sin(2 * np.pi * 2 * f0 * np.arange(0, n_samples))
x_ref += 0.1 * np.sin(2 * np.pi * 3 * f0 * np.arange(0, n_samples))

# Compute a test stft and stft magnitude

n_fft = 2048
hop_size = int(n_fft // 4)
window = np.hanning(n_fft)

spectro_ref = test_stft(x_ref, window, hop_size)
spectro_mag = np.absolute(spectro_ref)

# Reconstruct x from the stft magnitude

x_reconstruct = gl.griffin_lim_reconstruct(
    100, spectro_mag, window, 1.5, hop_size)

sf.write('x_ref.wav', x_ref * 0.5, int(fe))
sf.write('x_reconstruct.wav', x_reconstruct * 0.5, int(fe))
