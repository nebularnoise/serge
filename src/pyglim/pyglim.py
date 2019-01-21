"""

A python wrapper around a C implementation of the Griffin-Lim algorithm.

Author : Martin Fouilleul

"""

import os as os
import platform
import numpy as np
import ctypes as C

package_directory = os.path.dirname(os.path.abspath(__file__))
shared_lib = os.path.join(package_directory, 'lib', 'libglim')

if platform.system() == 'Linux':
	shared_lib = shared_lib + '.so'
elif platform.system() == 'Darwin':
	shared_lib = shared_lib + '.dylib'

libglim = C.cdll.LoadLibrary(shared_lib)

libglim.GriffinLimReconstruct.argtypes = (C.c_int, C.c_int, C.c_int, C.c_int, np.ctypeslib.ndpointer(
    dtype=np.float64), C.c_double, np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64))

libglim.GriffinLimReconstructFloat.argtypes = (C.c_int, C.c_int, C.c_int, C.c_int, np.ctypeslib.ndpointer(
    dtype=np.float32), C.c_float, np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32))


def griffin_lim_reconstruct(iteration_count, spectrogram_mag, window, window_gain, hop_size):
    """Griffin-Lim Reconstruction

    Reconstructs a signal from a magnitude spectrogram. Uses double precision floating point numbers

    Parameters
    ----------
    iteration_count: integer
	Number of iterations of the Griffin-Lim algorithm

    spectrogram_mag: list or numpy array
        Input magnitude spectrogram. The rows are consecutive time slices, the columns are the fft frequency bins (up to fft_size/2 + 1)

    window: list of numpy array
        Window to use during analysis and overlap-add reconstruction

    window_gain: float
        Gain resulting from applying a weighted overlap add of the window at the specified hop size

    hop_size: integer
	Number of samples between two consecutive time slices

    Returns
    -------
    numpy array
        Reconstructed signal
    """

    fft_size = len(window)
    slice_count = spectrogram_mag.shape[0]

    spectrogram_buffer = np.ascontiguousarray(spectrogram_mag, dtype=np.float64)
    window_buffer = np.ascontiguousarray(window, dtype=np.float64)
    signal_buffer = np.empty(
        (slice_count - 1) * hop_size + fft_size, dtype=np.float64)

    libglim.GriffinLimReconstruct(C.c_int(iteration_count), C.c_int(fft_size), C.c_int(hop_size), C.c_int(
        slice_count), window_buffer, C.c_double(window_gain), spectrogram_buffer, signal_buffer)

    return signal_buffer


def griffin_lim_reconstruct_single_precision(iteration_count, spectrogram_mag, window, window_gain, hop_size):
    """Griffin-Lim Reconstruction

    Reconstructs a signal from a magnitude spectrogram. This version uses single precision floating point numbers

    Parameters
    ----------
    iteration_count: integer
	Number of iterations of the Griffin-Lim algorithm

    spectrogram_mag: list or numpy array
        Input magnitude spectrogram. The rows are consecutive time slices, the columns are the fft frequency bins (up to fft_size/2 + 1)

    window: list of numpy array
        Window to use during analysis and overlap-add reconstruction

    window_gain: float
        Gain resulting from applying a weighted overlap add of the window at the specified hop size

    hop_size: integer
	Number of samples between two consecutive time slices

    Returns
    -------
    numpy array
        Reconstructed signal
    """

    fft_size = len(window)
    slice_count = spectrogram_mag.shape[0]

    spectrogram_buffer = np.ascontiguousarray(spectrogram_mag, dtype=np.float32)
    window_buffer = np.ascontiguousarray(window, dtype=np.float32)
    signal_buffer = np.empty(
        (slice_count - 1) * hop_size + fft_size, dtype=np.float32)

    libglim.GriffinLimReconstructFloat(C.c_int(iteration_count), C.c_int(fft_size), C.c_int(hop_size), C.c_int(
        slice_count), window_buffer, C.c_float(window_gain), spectrogram_buffer, signal_buffer)

    return signal_buffer
