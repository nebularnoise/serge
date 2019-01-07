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
#include"fftw3.h"
#include"griffin_lim.h"

void GriffinLimSTFT(fftwf_plan plan, int fftSize, int hopSize, int sliceCount, float* window, float* signalIn, fftwf_complex* spectrogramOut, float* fftAlignedBuffer)
{
	//NOTE(martin): spectrogramOut and fftAlignedIn must have been allocated with fftwf_malloc() to ensure an alignement suitable to simd instructions

	int start = 0;
	int end = hopSize;
	int sliceStart = 0;

	for(int slice=0; slice<sliceCount; slice++)
	{
		for(int i=0; i<fftSize; i++)
		{
			fftAlignedBuffer[i] = signalIn[start+i]*window[i];
		}
		fftwf_execute_dft_r2c(plan, fftAlignedBuffer, spectrogramOut + sliceStart);

		sliceStart += fftSize;
		start += hopSize;
		end += hopSize;
	}
}

void GriffinLimISTFT(fftwf_plan plan, int fftSize, int hopSize, int sliceCount, float* window, fftwf_complex* spectrogramIn, float* signalOut, float* fftAlignedBuffer)
{
	int start = 0;
	int end = hopSize;
	int sliceStart = 0;

	for(int slice=0; slice<sliceCount; slice++)
	{
		fftwf_execute_dft_c2r(plan, spectrogramIn + sliceStart, fftAlignedBuffer);

		for(int i=0; i<fftSize; i++)
		{
			signalOut[start + i] += fftAlignedBuffer[i]*window[i];
		}

		sliceStart += fftSize;
		start += hopSize;
		end += hopSize;
	}
}


void GriffinLimReconstruct(int iterCount, int fftSize, int hopSize, int sliceCount, float* window, float* magSpectrogramIn, float* signalOut)
{
	int sampleCount = sliceCount * hopSize + fftSize;
	int spectrogramFloatsSize = sliceCount*(fftSize/2+1)*2;

	fftwf_complex* spectrogramEstimate = (fftwf_complex*)fftwf_malloc((fftSize/2+1) * sliceCount * sizeof(fftwf_complex));
	float* signalWork = (float*)fftwf_malloc(sampleCount*sizeof(float));
	float* fftAlignedBuffer = (float*)fftwf_malloc(fftSize*sizeof(float));

	fftwf_plan forwardPlan = fftwf_plan_dft_r2c_1d(fftSize, fftAlignedBuffer, spectrogramEstimate, FFTW_ESTIMATE);
	fftwf_plan backwardPlan = fftwf_plan_dft_c2r_1d(fftSize, spectrogramEstimate, fftAlignedBuffer, FFTW_ESTIMATE);

	float* eulers = (float*)alloca(sliceCount * fftSize * 2 * sizeof(float));

	//NOTE(martin): initialize with random phases
	for(int i=0; i< spectrogramFloatsSize; i += 2)
	{
		float angle = (rand()/(float)RAND_MAX)*2*M_PI - M_PI;
		eulers[i] = cos(angle);
		eulers[i+1] = sin(angle);
	}

	for(int it = 0; it < iterCount; it++)
	{
		//NOTE(martin): compute spectrum estimate
		for(int i = 0; i < spectrogramFloatsSize; i += 2)
		{
			((float*)spectrogramEstimate)[i] = magSpectrogramIn[i] * eulers[i];
			((float*)spectrogramEstimate)[i+1] = magSpectrogramIn[i+1] * eulers[i+1];
		}

		GriffinLimISTFT(backwardPlan, fftSize, hopSize, sliceCount, window, spectrogramEstimate, signalWork, fftAlignedBuffer);
		GriffinLimSTFT(forwardPlan, fftSize, hopSize, sliceCount, window, signalWork, spectrogramEstimate, fftAlignedBuffer);

		for(int i=0; i < spectrogramFloatsSize; i += 2)
		{
			float angle = atan(((float*)spectrogramEstimate)[i+1]/((float*)spectrogramEstimate)[i]);
			eulers[i] = cos(angle);
			eulers[i+1] = sin(angle);
		}
	}

	//NOTE(martin): copy output buffer and clean

	memcpy(signalOut, signalWork, sampleCount*sizeof(float));

	fftwf_destroy_plan(forwardPlan);
	fftwf_destroy_plan(backwardPlan);

	fftwf_free(fftAlignedBuffer);
	fftwf_free(signalWork);
	fftwf_free(spectrogramEstimate);
}
