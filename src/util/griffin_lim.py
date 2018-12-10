import numpy as np


def griffin_lim_stft(x, window, hop_size):

    n_fft = len(window)
    n_slices = int(np.floor((len(x) - len(window)) / hop_size))

    spectrum = np.empty((n_fft / 2 + 1, n_slices), dtype=complex)

    start_index = 0
    end_index = n_fft

    for k in range(0, n_slices):
        xw = x[start_index:end_index] * window
        spectrum[:, k] = np.fft.rfft(xw, n_fft)

        start_index += hop_size
        end_index += hop_size

    return spectrum


def griffin_lim_istft(s, window, hop_size):

    n_fft = len(window)
    n_slices = s.shape[1]
    n_samples = int(n_slices * hop_size + n_fft)

    start_index = 0
    end_index = n_fft

    x = np.zeros((n_samples))

    for k in range(0, n_slices):
        xw = np.real(np.fft.irfft(s[:, k])) * window
        x[start_index:end_index] += xw

        start_index += hop_size
        end_index += hop_size

    return x


def griffin_lim_reconstruct(spectrum_mag, window, hop_size, nIter):

    maxIterations = nIter

    n_slices = len(spectrum_mag[0])
    n_samples = n_slices * hop_size + len(window)

    angles = np.exp(2j * np.pi * np.random.rand(*spectrum_mag.shape))

    x = np.random.randn(n_samples)

    for k in range(0, maxIterations):
	spectrum_est = spectrum_mag * angles
	x = griffin_lim_istft(spectrum_est, window, hop_size)
	spectrum_reconstruct = griffin_lim_stft(x, window, hop_size)
        angles = np.exp(1j * np.angle(spectrum_reconstruct))

    return x
