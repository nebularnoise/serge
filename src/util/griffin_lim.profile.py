# author: Martin Fouilleul

import numpy as np
import griffin_lim as gl
import time as tm

# Build test signal

fe = 44100.
length_seconds = 0.5
n_samples = int(fe * length_seconds)

f0 = 440. / fe

x_ref = 0.5 * np.sin(2 * np.pi * f0 * np.arange(0, n_samples))
x_ref += 0.1 * np.sin(2 * np.pi * 2 * f0 * np.arange(0, n_samples))
x_ref += 0.1 * np.sin(2 * np.pi * 3 * f0 * np.arange(0, n_samples))

# Compute stft and stft magnitude

n_fft = 2048
hop_size = int(n_fft // 4)
window = np.hanning(n_fft)

spectrum_ref = gl.griffin_lim_stft(x_ref, window, hop_size)
spectrum_mag = np.absolute(spectrum_ref)

# Reconstruct x from the stft magnitude with both au and our custom code

start = tm.perf_counter()
x_reconstruct = gl.griffin_lim_reconstruct(
    spectrum_mag, window, hop_size, 100)
custom_gl_time = tm.perf_counter() - start

# Print times :

print("time : " + str(custom_gl_time))
