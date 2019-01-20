/*************************************************************//**
*
*	@file	griffin_lim_generic.cpp
*	@date	07/01/2019
*	@author Martin Fouilleul
*
*****************************************************************/
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include"fftw3.h"

//NOTE(martin): This generic file is meant to be included after some #define directive to set the floating point precision
//		You must #define FP_SINGLE_PRECISION to enable float, otherwise precision defaults to double


#ifndef FP_SINGLE_PRECISION
	#define GRIFFIN_LIM_STFT GriffinLimSTFT
	#define GRIFFIN_LIM_ISTFT GriffinLimISTFT
	#define GRIFFIN_LIM_RECONSTRUCT GriffinLimReconstruct

	#define FLOAT_VAL double
	#define FFTW_COMPLEX fftw_complex
	#define FFTW_MALLOC fftw_malloc
	#define FFTW_FREE fftw_free
	#define FFTW_PLAN fftw_plan
	#define FFTW_DESTROY_PLAN fftw_destroy_plan
	#define FFTW_PLAN_DFT_R2C_1D fftw_plan_dft_r2c_1d
	#define FFTW_PLAN_DFT_C2R_1D fftw_plan_dft_c2r_1d
	#define FFTW_EXECUTE_DFT_R2C fftw_execute_dft_r2c
	#define FFTW_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#else
	#define GRIFFIN_LIM_STFT GriffinLimSTFTFloat
	#define GRIFFIN_LIM_ISTFT GriffinLimISTFTFloat
	#define GRIFFIN_LIM_RECONSTRUCT GriffinLimReconstructFloat

	#define FLOAT_VAL float
	#define FFTW_COMPLEX fftwf_complex
	#define FFTW_MALLOC fftwf_malloc
	#define FFTW_FREE fftwf_free
	#define FFTW_PLAN fftwf_plan
	#define FFTW_DESTROY_PLAN fftwf_destroy_plan
	#define FFTW_PLAN_DFT_R2C_1D fftwf_plan_dft_r2c_1d
	#define FFTW_PLAN_DFT_C2R_1D fftwf_plan_dft_c2r_1d
	#define FFTW_EXECUTE_DFT_R2C fftwf_execute_dft_r2c
	#define FFTW_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#endif

/**
	@brief Short-Term Fourier Transform

	@param plan The FFTW plan used to compute Fast Fourier Transforms
	@param fftSize Logical size of the DFT
	@param hopSize Hop size between two consecutive time slices
	@param sliceCount Number of slices in the STFT
	@param window Analysis window, of size (fftSize)
	@param signalIn Input signal, of size ((sliceCount-1)*hopSize + fftSize)
	@param spectrogramOut Output spectrogram, of size (sliceCount*(fftSize/2 + 1))
	@param fftScratchBuffer Work buffer used by the FFTW functions, of size (fftSize)
*/
void GRIFFIN_LIM_STFT(FFTW_PLAN plan,
		    int fftSize,
		    int hopSize,
		    int sliceCount,
		    FLOAT_VAL* window,
		    FLOAT_VAL* signalIn,
		    FFTW_COMPLEX* spectrogramOut,
		    FLOAT_VAL* fftScratchBuffer)
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
		FFTW_EXECUTE_DFT_R2C(plan, fftScratchBuffer, spectrogramOut + sliceStart);

		sliceStart += (fftSize/2+1);
		start += hopSize;
	}
}

/**
	@brief Inverse Short-Term Fourier Transform

	@param plan The FFTW plan used to compute Inverse Fast Fourier Transforms
	@param fftSize Logical size of the DFT
	@param hopSize Hop size between two consecutive time slices
	@param sliceCount Number of slices in the STFT
	@param window Analysis window, of size (fftSize)
	@param spectrogramIn Output spectrogram, of size (sliceCount*(fftSize/2 + 1))
	@param signalOut Input signal, of size ((sliceCount-1)*hopSize + fftSize)
	@param fftScratchBuffer Work buffer used by the FFTW functions, of size (fftSize)
*/

void GRIFFIN_LIM_ISTFT(FFTW_PLAN plan,
		     int fftSize,
		     int hopSize,
		     int sliceCount,
		     FLOAT_VAL* window,
		     FLOAT_VAL windowGain,
		     FFTW_COMPLEX* spectrogramIn,
		     FLOAT_VAL* signalOut,
		     FLOAT_VAL* fftScratchBuffer)
{
	int start = 0;
	int sliceStart = 0;

	FLOAT_VAL normalize = 1./(windowGain*fftSize);

	memset(signalOut, 0, ((sliceCount-1)*hopSize + fftSize)*sizeof(FLOAT_VAL));

	for(int slice=0; slice<sliceCount; slice++)
	{
		memset(fftScratchBuffer, 0, fftSize*sizeof(FLOAT_VAL));

		FFTW_EXECUTE_DFT_C2R(plan, spectrogramIn + sliceStart, fftScratchBuffer);

		for(int i=0; i<fftSize; i++)
		{
			signalOut[start + i] += normalize*fftScratchBuffer[i]*window[i];
		}

		sliceStart += (fftSize/2+1);
		start += hopSize;
	}
}

/**
	@brief Implements the Griffin-Lim algorithm for reconstructing a signal from a magnitude spectrogram.

	@param iterCount	The number of iterations of the algorithm
	@param fftSize		The logical size of the DFT, which is also the size of the window
	@param hopSize		Number of samples between to consecutive DFT slices
	@param sliceCount	Number of DFT slices in the spectrogram
	@param window		The window function, of size (fftSize)
	@param windowGain	The gain resulting from overlap-adding the squared window (this is used for gain compensation)
	@param magSpectrogram	The input magnitude spectrogram, of dimension (sliceCount , (fftSize/2+1)). The rows are the DFT slices, the columns are the (fftSize/2+1) DFT bins of a real valued signal.
	@param signal		The output estimated signal, of size ((sliceCount-1)*hopSize + fftSize)

*/

extern "C" void GRIFFIN_LIM_RECONSTRUCT(int iterCount,
				      int fftSize,
				      int hopSize,
				      int sliceCount,
				      FLOAT_VAL* window,
				      FLOAT_VAL windowGain,
				      FLOAT_VAL* magSpectrogram,
				      FLOAT_VAL* signal)
{
	int sampleCount = (sliceCount - 1) * hopSize + fftSize;

	FFTW_COMPLEX* spectrogramEstimate = (FFTW_COMPLEX*)fftwf_malloc((fftSize/2+1) * sliceCount * sizeof(FFTW_COMPLEX));
	FLOAT_VAL* signalWork = (FLOAT_VAL*)fftwf_malloc(sampleCount*sizeof(FLOAT_VAL));
	FLOAT_VAL* fftScratchBuffer = (FLOAT_VAL*)fftwf_malloc(fftSize*sizeof(FLOAT_VAL));

	FFTW_PLAN forwardPlan = FFTW_PLAN_DFT_R2C_1D(fftSize, fftScratchBuffer, spectrogramEstimate, FFTW_ESTIMATE);
	FFTW_PLAN backwardPlan = FFTW_PLAN_DFT_C2R_1D(fftSize, spectrogramEstimate, fftScratchBuffer, FFTW_ESTIMATE);

	//NOTE(martin): initalize signal with random values
	srand(54652);
	for(int i=0; i<sampleCount; i++)
	{
		signalWork[i] = rand()/(FLOAT_VAL)RAND_MAX;
	}

	for(int it = 0; it < iterCount; it++)
	{
		GRIFFIN_LIM_STFT(forwardPlan, fftSize, hopSize, sliceCount, window, signalWork, spectrogramEstimate, fftScratchBuffer);
		for(int i=0; i<(fftSize/2+1)*sliceCount; i++)
		{
			FLOAT_VAL mag = magSpectrogram[i];
			FLOAT_VAL re = spectrogramEstimate[i][0];
			FLOAT_VAL im = spectrogramEstimate[i][1];
			if((im == 0) && (re == 0))
			{
				spectrogramEstimate[i][0] = mag;
				spectrogramEstimate[i][1] = 0;
			}
			else
			{
				FLOAT_VAL angle = atan(im/re);
				if(re < 0)
				{
					angle += M_PI;
				}

				spectrogramEstimate[i][0] = mag * cos(angle);
				spectrogramEstimate[i][1] = mag * sin(angle);
			}
		}
		GRIFFIN_LIM_ISTFT(backwardPlan, fftSize, hopSize, sliceCount, window, windowGain, spectrogramEstimate, signalWork, fftScratchBuffer);
	}

	//NOTE(martin): copy output buffer and clean

	memcpy(signal, signalWork, sampleCount*sizeof(FLOAT_VAL));

	FFTW_DESTROY_PLAN(forwardPlan);
	FFTW_DESTROY_PLAN(backwardPlan);

	fftwf_free(fftScratchBuffer);
	fftwf_free(signalWork);
	fftwf_free(spectrogramEstimate);
}
