import numpy as np
import griffin_lim as gl
from matplotlib import pyplot as plt

# Build test signal

fe = 44100.
length_seconds = 0.5
n_samples = int(fe * length_seconds)

f0 = 440. / fe

x_ref = np.sin(2 * np.pi * f0 * np.arange(0, n_samples))
x_ref += np.sin(2 * np.pi * 2.5 * f0 * np.arange(0, n_samples) + np.pi / 2)
x_ref += np.sin(2 * np.pi * 3.2 * f0 * np.arange(0, n_samples) + np.pi / 3)

# Compute stft and stft magnitude

n_fft = 2048
hop_size = int(np.floor(n_fft / 2))
window = np.sqrt(np.hanning(n_fft))

spectrum_ref = gl.griffin_lim_stft(x_ref, window, hop_size)
spectrum_mag = np.absolute(spectrum_ref)

x_reconstruct_ref = gl.griffin_lim_istft(spectrum_ref, window, hop_size)

# Reconstruct x from the stft magnitude

x_reconstruct = gl.griffin_lim_reconstruct(
    spectrum_mag, window, hop_size, 1000)

# Plot the reconstruction against the reference

plt.plot(x_ref, 'r')
plt.plot(x_reconstruct_ref, 'g')
plt.plot(x_reconstruct, 'b')
plt.show()
