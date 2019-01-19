# author: Martin Fouilleul
import numpy as np


def griffin_lim_stft(x, window, hop_size):
    """STFT

    Compute the short-term Fourier transform of the input signal

    Parameters
    ----------
    x: numpy array
        The input signal
    window: numpy array
        The window to use during analysis
    hop_size: integer
        The hop size between two consecutive slices

    Returns
    -------
    numpy array
        The short-time Fourier transform of the signal x
    """

    n_fft = len(window)
    spectrum = np.array([np.fft.rfft(window*x[i:i+n_fft])
                         for i in range(0, len(x)-n_fft, hop_size)])

    return spectrum


def griffin_lim_istft(s, window, hop_size):
    """ISTFT

    Inverts a full short-time Fourier transform

    Parameters
    ----------
    s: numpy array
        Complex spectrogram. The lines are the slices, and the columns are the frequency bins.
    window: numpy array
        The window to use during overlap-add reconstruction
    hop_size: integer
        The hop size between two consecutive slices

    Returns
    -------
    numpy array
        The reconstructed signal
    """

    n_fft = len(window)
    n_samples = int(s.shape[0] * hop_size + n_fft)

    x = np.zeros(n_samples)
    for n, i in enumerate(range(0, len(x)-n_fft, hop_size)):
        x[i:i+n_fft] += window*np.real(np.fft.irfft(s[n]))

    return x


def griffin_lim_reconstruct(spectrum_mag, window, hop_size, nIter):
    """Griffin-Lim Algorithm

    Reconstructs a signal from a magnitude spectrogram

    Parameters
    ----------
    spectrum_mag: numpy array
        Magnitude spectrogram. The lines are the slices, and the columns are the frequency bins.
    window: numpy array
        The window to use during overlap-add analysis/reconstruction
    hop_size: integer
        The hop size between two consecutive slices
    nIter: integer
        The number of iterations to apply

    Returns
    -------
    numpy array
        The reconstructed signal
    """

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
