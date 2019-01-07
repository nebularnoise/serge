import numpy as np

def griffin_lim_stft(x, window, hop_size):

    n_fft = len(window)
    spectrum =  np.array([np.fft.rfft(window*x[i:i+n_fft])
                     for i in range(0, len(x)-n_fft, hop_size)])

    return spectrum


def griffin_lim_istft(s, window, hop_size):

    n_fft = len(window)
    n_samples = int(s.shape[0] * hop_size + n_fft)

    x = np.zeros(n_samples)
    for n,i in enumerate(range(0, len(x)-n_fft, hop_size)):
        x[i:i+n_fft] += window*np.real(np.fft.irfft(s[n]))

    return x


def griffin_lim_reconstruct(spectrum_mag, window, hop_size, nIter):

    maxIterations = nIter

    n_slices = spectrum_mag.shape[0]
    n_samples = n_slices * hop_size + len(window)

    angles = np.exp(2j * np.pi * np.random.rand(*spectrum_mag.shape))

    x = np.random.randn(n_samples)

    for k in range(0, maxIterations):
        spectrum_est = spectrum_mag * angles
        x = griffin_lim_istft(spectrum_est, window, hop_size)
        spectrum_reconstruct = griffin_lim_stft(x, window, hop_size)
        angles = np.exp(1j * np.angle(spectrum_reconstruct))

    return x
