import numpy as np
import griffin_lim as gl
from matplotlib import pyplot as plt
import soundfile as sf
import audio_utilities as au

# Build test signal

fe = 44100.
length_seconds = 0.5
n_samples = int(fe * length_seconds)

f0 = 440. / fe

x_ref = 0.5 * np.sin(2 * np.pi * f0 * np.arange(0, n_samples))
x_ref += 0.1* np.sin(2 * np.pi * 2 * f0 * np.arange(0, n_samples))
x_ref += 0.1 * np.sin(2 * np.pi * 3 * f0 * np.arange(0, n_samples))

# Compute stft and stft magnitude

n_fft = 2048
hop_size = int(n_fft // 4)
window = np.hanning(n_fft)

spectrum_ref = gl.griffin_lim_stft(x_ref, window, hop_size)
spectrum_mag = np.absolute(spectrum_ref)

x_reconstruct_ref = gl.griffin_lim_istft(spectrum_ref, window, hop_size)

plt.imshow(np.transpose(spectrum_mag), origin='lower', extent=(0, length_seconds * 2000, 0, n_fft/2 * 4))
plt.show()

# Reconstruct x from the stft magnitude

x_reconstruct = gl.griffin_lim_reconstruct(
    spectrum_mag, window, hop_size, 100)

x_reconstruct_au = au.reconstruct_signal_griffin_lim(spectrum_mag, n_fft, hop_size, 100)

s_reconstruct = gl.griffin_lim_stft(x_reconstruct, window, hop_size)

sf.write('x_ref.wav', x_reconstruct_ref * 0.5, int(fe))
sf.write('x_reconstruct.wav', x_reconstruct * 0.5, int(fe))
sf.write('x_reconstruct_au.wav', x_reconstruct_au * 0.5, int(fe))

# Plot the reconstruction against the reference

plt.plot(x_ref, 'r')
plt.plot(x_reconstruct_ref, 'g')
plt.plot(x_reconstruct, 'b')

plt.figure()
plt.imshow(np.absolute(np.transpose(s_reconstruct)), origin='lower', extent=(0, length_seconds * 2000, 0, n_fft/2  * 4))
plt.show()
