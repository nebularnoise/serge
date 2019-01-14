//*****************************************************************
//
//	$file: griffin_lim.cpp $
//	$date: 07/01/2019 $
//	$revision: $
//
//*****************************************************************
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include"fftw3.h"

void GriffinLimSTFT(fftwf_plan plan,
		    int fftSize,
		    int hopSize,
		    int sliceCount,
		    float* window,
		    float* signalIn,
		    fftwf_complex* spectrogramOut,
		    float* fftScratchBuffer)
{
	//NOTE(martin): spectrogramOut and fftAlignedIn must have been allocated with fftwf_malloc() to ensure an alignement suitable to simd instructions

	int start = 0;
	int sliceStart = 0;

	for(int slice=0; slice<sliceCount; slice++)
	{
		for(int i=0; i<fftSize; i++)
		{
			fftScratchBuffer[i] = signalIn[start+i]*window[i];
		}
		fftwf_execute_dft_r2c(plan, fftScratchBuffer, spectrogramOut + sliceStart);

		sliceStart += (fftSize/2+1);
		start += hopSize;
	}
}

void GriffinLimISTFT(fftwf_plan plan,
		     int fftSize,
		     int hopSize,
		     int sliceCount,
		     float* window,
		     float windowGain,
		     fftwf_complex* spectrogramIn,
		     float* signalOut,
		     float* fftScratchBuffer)
{
	int start = 0;
	int sliceStart = 0;

	float normalize = 1./(windowGain*fftSize);

	memset(signalOut, 0, ((sliceCount-1)*hopSize + fftSize)*sizeof(float));

	for(int slice=0; slice<sliceCount; slice++)
	{
		memset(fftScratchBuffer, 0, fftSize*sizeof(float));

		fftwf_execute_dft_c2r(plan, spectrogramIn + sliceStart, fftScratchBuffer);

		for(int i=0; i<fftSize; i++)
		{
			signalOut[start + i] += normalize*fftScratchBuffer[i]*window[i];
		}

		sliceStart += (fftSize/2+1);
		start += hopSize;
	}
}

extern "C" void GriffinLimReconstruct(int iterCount,
				      int fftSize,
				      int hopSize,
				      int sliceCount,
				      float* window,
				      float windowGain,
				      float* magSpectrogram,
				      float* signal)
{
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
		signal :	The output estimated signal, of size ((sliceCount-1)*hopSize + fftSize)

	*/
	int sampleCount = (sliceCount - 1) * hopSize + fftSize;

	fftwf_complex* spectrogramEstimate = (fftwf_complex*)fftwf_malloc((fftSize/2+1) * sliceCount * sizeof(fftwf_complex));
	float* signalWork = (float*)fftwf_malloc(sampleCount*sizeof(float));
	float* fftScratchBuffer = (float*)fftwf_malloc(fftSize*sizeof(float));

	fftwf_plan forwardPlan = fftwf_plan_dft_r2c_1d(fftSize, fftScratchBuffer, spectrogramEstimate, FFTW_ESTIMATE);
	fftwf_plan backwardPlan = fftwf_plan_dft_c2r_1d(fftSize, spectrogramEstimate, fftScratchBuffer, FFTW_ESTIMATE);

	//NOTE(martin): initalize signal with random values
	srand(54652);
	for(int i=0; i<sampleCount; i++)
	{
		signalWork[i] = rand()/(float)RAND_MAX;
	}

	for(int it = 0; it < iterCount; it++)
	{
		GriffinLimSTFT(forwardPlan, fftSize, hopSize, sliceCount, window, signalWork, spectrogramEstimate, fftScratchBuffer);
		for(int i=0; i<(fftSize/2+1)*sliceCount; i++)
		{
			float mag = magSpectrogram[i];
			float re = spectrogramEstimate[i][0];
			float im = spectrogramEstimate[i][1];
			if((im == 0) && (re == 0))
			{
				spectrogramEstimate[i][0] = mag;
				spectrogramEstimate[i][1] = 0;
			}
			else
			{
				float angle = atan(im/re);
				if(re < 0)
				{
					angle += M_PI;
				}

				spectrogramEstimate[i][0] = mag * cos(angle);
				spectrogramEstimate[i][1] = mag * sin(angle);
			}
		}
		GriffinLimISTFT(backwardPlan, fftSize, hopSize, sliceCount, window, windowGain, spectrogramEstimate, signalWork, fftScratchBuffer);
	}

	//NOTE(martin): copy output buffer and clean

	memcpy(signal, signalWork, sampleCount*sizeof(float));

	fftwf_destroy_plan(forwardPlan);
	fftwf_destroy_plan(backwardPlan);

	fftwf_free(fftScratchBuffer);
	fftwf_free(signalWork);
	fftwf_free(spectrogramEstimate);
}
