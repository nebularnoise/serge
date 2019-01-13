#include<stdio.h>
#include<math.h>
#include<sndfile.h>
#include"griffin_lim.cpp"


#define Square(a) ((a)*(a))
#define Magnitude(S) (sqrt(Square((S)[0])+Square((S)[1])))

void Hann(int count, float* windowOut)
{
	float invCount = 1./(count-1);
	for(int i=0; i<count; i++)
	{
		windowOut[i] = 0.5*(1 - cos(2*M_PI*i*invCount));
	}
}

int main(int argc, char** argv)
{
	const int fftSize = 2048;
	const int hopSize = fftSize/4;

	const float nSeconds = 0.5;
	const float sampleRate = 44100;

	const int sampleCount = ceil(nSeconds*sampleRate);
	const int sliceCount = floor((sampleCount - fftSize)/ (float)hopSize);

	float S[(fftSize/2+1)*sliceCount*2];
	float M[(fftSize/2+1)*sliceCount];

	float window[fftSize];
	float x[sampleCount];

	Hann(fftSize, window);

	//NOTE(martin): Generate test signal

	const int nFreqs = 3;
	float freq[nFreqs] = { 440, 880, 1320 };

	for(int i=0; i<sampleCount; i++)
	{
		x[i] = 0;
		x[i] += 0.5*sin( 2*M_PI*freq[0]*i/(float)sampleRate);
		x[i] += 0.1*sin( 2*M_PI*freq[1]*i/(float)sampleRate);
		x[i] += 0.1*sin( 2*M_PI*freq[2]*i/(float)sampleRate);
	}

	//NOTE(martin): Compute the stft of our test signal

	float*	fftwSignal = (float*)fftwf_malloc(fftSize*sizeof(float));
	fftwf_complex* fftwSpec = (fftwf_complex*)fftwf_malloc((fftSize/2+1)*sizeof(fftwf_complex));
	fftwf_plan p = fftwf_plan_dft_r2c_1d(fftSize, fftwSignal, fftwSpec, FFTW_ESTIMATE | FFTW_UNALIGNED);

	int start = 0;
	int sliceStart = 0;

	for(int slice=0; slice<sliceCount; slice++)
	{
		for(int i=0; i<fftSize; i++)
		{
			fftwSignal[i] = x[start+i]*window[i];
		}
		fftwf_execute_dft_r2c(p, fftwSignal, (fftwf_complex*)(S + sliceStart));

		sliceStart += (fftSize/2+1)*2;
		start += hopSize;
	}
	fftwf_destroy_plan(p);

	//NOTE(martin): compute magnitude spectrogram
	for(int i=0; i<(fftSize/2+1)*sliceCount; i++)
	{
		M[i] = Magnitude(S + 2*i);
	}

	//NOTE(martin): write test signal to file
	SF_INFO sfInfo;
	memset(&sfInfo, 0, sizeof(sfInfo));
	sfInfo.samplerate = 44100;
	sfInfo.channels = 1;
	sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

	SNDFILE* sndFile = sf_open("ref_x.wav", SFM_WRITE, &sfInfo);
	sf_write_float(sndFile, x, sampleCount);
	sf_close(sndFile);

	//NOTE(martin): reconstruct the signal from magnitude spectrogram
	memset(x, 0, sampleCount*sizeof(float));
	GriffinLimReconstruct(200, fftSize, hopSize, sliceCount, window, 1.5, M, x);

	//NOTE(martin): write reconstructed signal to file
	sndFile = sf_open("reconstructed_x.wav", SFM_WRITE, &sfInfo);
	sf_write_float(sndFile, x, sampleCount);
	sf_close(sndFile);

	return(0);
}
