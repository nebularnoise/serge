//*****************************************************************
//
//	$file: griffin_lim.h $
//	$date: 07/01/2019 $
//	$revision: $
//
//*****************************************************************
#ifndef _GRIFFIN_LIM_H_
#define _GRIFFIN_LIM_H_

/*NOTE(martin)
	Implements the Griffin-Lim algorithm for reconstructing a signal from a magnitude spectrogram.

	iterCount :	The number of iterations of the algorithm
	fftSize :	The logical size of the DFT, which is also the size of the window
	hopSize :	Number of samples between to consecutive DFT slices
	sliceCount :	Number of DFT slices in the spectrogram
	window :	The window function, of size (fftSize)
	windowGain :	The gain resulting from overlap-adding the squared window (this is used for gain compensation)
	magSpectrogram : The input magnitude spectrogram, of dimension (sliceCount , (fftSize/2+1)). The rows are the DFT slices,
			  the columns are the (fftSize/2+1) DFT bins of a real valued signal.
	signal :	The output estimated signal, of size (sliceCount*hopSize + fftSize)

*/

void GriffinLimReconstruct(int iterCount,
			   int fftSize,
			   int hopSize,
			   int sliceCount,
			   float* window,
			   float windowGain,
			   float* magSpectrogram,
			   float* signal);

#endif // _GRIFFIN_LIM_H_
