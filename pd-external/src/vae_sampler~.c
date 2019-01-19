/*************************************************************//**
*
*	@file	vae_sampler~.c
*	@date	28/10/2018
*	@author Martin Fouilleul
*
*****************************************************************/
#include<stdlib.h>	// malloc
#include<string.h>	// memset
#include<math.h>	// M_PI, cos, ...
#include<stdio.h>	// printf
#include<pthread.h>
#include"m_pd.h"
#include"profile.h"
#include"vae_util.h"
#include"griffin_lim.h"

//-----------------------------------------------------------------
// debug printing macros
//-----------------------------------------------------------------

//NOTE(martin): we provide DEBUG_PRINTF because pd's post methods seems to be thread-unsafe
//		whereas printf is required to be thread safe by POSIX

#ifdef DEBUG
	#define DEBUG_POST(s, ...) post(s, ##__VA_ARGS__)
	#define DEBUG_PRINTF(s, ...) printf(s, ##__VA_ARGS__)
#else
	#define DEBUG_POST(s, ...)
	#define DEBUG_PRINTF(s, ...)
#endif
#define ERROR_POST(s, ...) error(s, ##__VA_ARGS__)
#define ERROR_PRINTF(s, ...) fprintf(stderr, s, ##__VA_ARGS__)
#define POST(s, ...) post(s, ##__VA_ARGS__)

//-----------------------------------------------------------------
// object definition
//-----------------------------------------------------------------

//TODO(martin): would be much safer with const int (esp. with respect to parenthesing)
//		but gcc on linux seems to fail when instancing buffers with constants declared in global scope...??

#ifdef DEBUG
#define GRIFFIN_LIM_ITERATION_COUNT 40
#else
#define GRIFFIN_LIM_ITERATION_COUNT 100
#endif

#define MODEL_SLICE_COUNT	128
#define MODEL_FFT_SIZE		2048
#define MODEL_BIN_COUNT		1025
#define MODEL_SPECTROGRAM_SIZE	(MODEL_SLICE_COUNT * MODEL_BIN_COUNT)
#define MODEL_HOP_SIZE		(MODEL_FFT_SIZE / 8)
#define MODEL_OLA_GAIN		3
#define SAMPLE_BUFFER_SIZE	((MODEL_SLICE_COUNT - 1) * MODEL_HOP_SIZE + MODEL_FFT_SIZE)

#define	VOICE_COUNT		4

#define GL_BATCH_SLICE_COUNT	32
#define GL_BATCH_COUNT		(MODEL_SLICE_COUNT / GL_BATCH_SLICE_COUNT)
#define GL_BATCH_SIZE		(GL_BATCH_SLICE_COUNT * MODEL_BIN_COUNT)
#define GL_BATCH_SAMPLES	((GL_BATCH_SLICE_COUNT-1) * MODEL_HOP_SIZE + MODEL_FFT_SIZE)
#define GL_BATCH_HOP		(MODEL_HOP_SIZE * GL_BATCH_SLICE_COUNT)


typedef int voice_head;
const voice_head HEAD_PLAY_FLAG    = 1<<31,
                 HEAD_COUNTER_MASK = ~(1<<31);

static float HANN_WINDOW[MODEL_FFT_SIZE];

typedef struct poly_voice_t
{
	volatile int	note;
	volatile int	stream;
	volatile voice_head	head;
	volatile int	endCursor;
	pthread_cond_t	condition;
	float c0, c1, c2, c3;

} poly_voice;

struct vae_sampler_t;

typedef struct worker_object_t
{
	struct vae_sampler_t* sampler;
	int	     voiceIndex;
} worker_object;

typedef struct vae_sampler_t
{
	t_object	obj;
	t_outlet*	out;

	vae_model*	model;

	int		nextVoice;
	float		spectrograms[VOICE_COUNT][MODEL_SPECTROGRAM_SIZE];
	float		buffers[VOICE_COUNT][SAMPLE_BUFFER_SIZE];
	poly_voice	voices[VOICE_COUNT];
	pthread_t	workers[VOICE_COUNT];
	worker_object	workerObjects[VOICE_COUNT];

} vae_sampler;

static	t_class* vae_sampler_class;

//-----------------------------------------------------------------
// worker threads
//-----------------------------------------------------------------

/**
	@brief Worker thread entry point
	@param x The address of a worker_object struct which contains the settings for the worker thread.
	@return 0
*/
void* StreamVoiceSamples(void* x)
{
	worker_object* object = (worker_object*)x;
	vae_sampler* sampler = object->sampler;
	int voiceIndex = object->voiceIndex;

	poly_voice* voice = &(sampler->voices[voiceIndex]);
	float* spectrogram = sampler->spectrograms[voiceIndex];
	float* samplesBuffer = &(sampler->buffers[voiceIndex][0]);

	float batchBuffer[GL_BATCH_SAMPLES];
	memset(batchBuffer, 0, GL_BATCH_SAMPLES*sizeof(float));

	pthread_mutex_t mutex;
	pthread_mutex_init(&mutex, 0);	//TODO(martin): where do we destroy this mutex if indeed we need to ?
	pthread_mutex_lock(&mutex);

	//TODO(martin): eventually, set an exit condition, unlock the mutexes and join the threads uppon exit,
	//		it's cleaner than cancelling in the main thread
	while(1)
	{
		//NOTE(martin): wait for our voice to be allocated and guard against spurious wakeups
		while(voice->stream == 0)
		{
			pthread_cond_wait(&(voice->condition), &mutex);
		}

		DEBUG_PRINTF("Wake up worker thread %i for note %i\n", voiceIndex, voice->note);

		memset(samplesBuffer, 0, SAMPLE_BUFFER_SIZE*sizeof(float));

		//NOTE(martin): get the spectrogram from our model
		int err = 0;
		TIME_BLOCK_START();
		err = VaeModelGetSpectrogram(sampler->model, MODEL_SPECTROGRAM_SIZE, spectrogram, voice->c0, voice->c1, voice->c2, voice->c3, voice->note);
		TIME_BLOCK_END("VaeModelGetSpectrogram()");

		if(err)
		{
			ERROR_PRINTF("Failed to get spectrogram from model ");
			switch(err)
			{
				case VAE_MODEL_NOT_LOADED:
					ERROR_PRINTF("(model not loaded)...\n");
					break;
				case VAE_MODEL_BAD_SIZE:
					ERROR_PRINTF("(bad tensor sizes)...\n");
					break;
				case VAE_MODEL_THROW:
					ERROR_PRINTF("(torch script exception)...\n");
					break;
			}

			//NOTE(martin): we exit early if the model could not load the samples. The dsp thread will read an empty frame
			//		so the voice stays busy, but exiting and deallocating the voice early would make the threading
			//		model more complex.
			voice->stream = 0;
			voice->endCursor = SAMPLE_BUFFER_SIZE+1;
			continue;
		}
		else
		{
			DEBUG_PRINTF("Got spectrogram from model");
		}

		//NOTE(martin): stream griffin lim batches

		int batchStart = 0;
		int samplesStart = 0;

		TIME_BLOCK_START();
		for(int i=0; i<GL_BATCH_COUNT; i++)
		{
			GriffinLimReconstruct(GRIFFIN_LIM_ITERATION_COUNT,
					      MODEL_FFT_SIZE,
					      MODEL_HOP_SIZE,
					      GL_BATCH_SLICE_COUNT,
					      HANN_WINDOW,
					      MODEL_OLA_GAIN,
					      (spectrogram+batchStart),
					      batchBuffer);

			float* dst = (samplesBuffer+samplesStart);
			for(int j=0; j<GL_BATCH_SAMPLES;j++)
			{
				dst[j] += batchBuffer[j];
			}
			samplesStart += GL_BATCH_HOP;
			batchStart += GL_BATCH_SIZE;
			voice->endCursor += GL_BATCH_HOP;
		}
		TIME_BLOCK_END("Griffin-Lim total");

		voice->stream = 0;
		voice->endCursor = SAMPLE_BUFFER_SIZE+1;
		DEBUG_PRINTF("Worker thread %i finished decoding note %i, go to sleep\n", voiceIndex, voice->note);
	}
	return(0);
}

//-----------------------------------------------------------------
// methods
//-----------------------------------------------------------------

/**
	@brief Generate a Hann window
	@param count size of the window.
	@param windowOut Output buffer which will hold the window. Must be of size (count)
*/
void Hann(int count, float* windowOut)
{
	float invCount = 1./(count-1);
	for(int i=0; i<count; i++)
	{
		windowOut[i] = 0.5*(1 - cos(2*M_PI*i*invCount));
	}
}

/**
	@brief The external perform routine

	This function reads the currently playing voices' buffers and mixes them to the external's outlet
*/
t_int* vae_sampler_perform(t_int* w)
{
	vae_sampler* x	= (vae_sampler*)w[1];
	t_sample* out	= (t_sample*)w[2];
	int n		= (int)w[3];

	memset(out, 0, n*sizeof(float));

	for(int voiceIndex = 0; voiceIndex < VOICE_COUNT; voiceIndex++)
	{
		float* buffer = x->buffers[voiceIndex];
		poly_voice* voice = &x->voices[voiceIndex];
		voice_head head = voice->head;
		int endCursor = voice->endCursor;

		if(head & HEAD_PLAY_FLAG)
		{
			int counter = head & HEAD_COUNTER_MASK;

			for(int i=0; (i<n) && (counter<endCursor); i++)
			{
				out[i] += buffer[counter];
				counter++;
			}
			#ifdef DEBUG
			if(  counter > 0
			  && counter < SAMPLE_BUFFER_SIZE
			  && counter >= endCursor)
			{
				DEBUG_POST("Warning : Griffin Lim worker thread underrun");
			}
			#endif

			if(counter >= SAMPLE_BUFFER_SIZE)
			{
				DEBUG_PRINTF("Perform routine finished playing voice %i\n", voiceIndex);

				voice->head = 0;
				voice->endCursor = 0;
				voice->note = 0;
			}
			else
			{
				x->voices[voiceIndex].head = HEAD_PLAY_FLAG | counter;
			}
		}
	}
	return(w+4);
}

void vae_sampler_dsp(vae_sampler* x, t_signal** sp)
{
	dsp_add(vae_sampler_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

void vae_sampler_load(vae_sampler* x, t_symbol* sym)
{
	if(VaeModelLoad(x->model, sym->s_name))
	{
		ERROR_POST("Can't load model %s...", sym->s_name);
	}
	else
	{
		POST("Loaded model %s", sym->s_name);
	}
}

/**
	@brief Allocate a voice and fires up its worker thread

	This function first determines if the requested note is already playing. If not, it tries to allocate a voice for this note, and wakes up its associated worker thread.
*/
void vae_sampler_fire(vae_sampler* x, t_symbol* sym, float fnote, float c0, float c1, float c2, float c3)
{
	int note = (int)floorf(fnote);

	//NOTE(martin): check if this note is already allocated
	for(int i=0; i<VOICE_COUNT; i++)
	{
		if(x->voices[i].note == note)
		{
			//NOTE: the given note is already playing... (we don't retrigger, but maybe we should ?)
			DEBUG_POST("warning: note %i is already playing", note);
			return;
		}
	}

	int voiceIndex = x->nextVoice;
	poly_voice* voice = &(x->voices[voiceIndex]);

	if(!voice->note)
	{
		//NOTE(martin): allocate the voice and wakeup worker thread

		DEBUG_POST("Play (%f %f %f %f), note %i", c0, c1, c2, c3, note);

		x->nextVoice++;
		if(x->nextVoice >= VOICE_COUNT)
		{
			x->nextVoice = 0;
		}
		voice->endCursor = 0;
		voice->note = note;

		voice->c0 = c0;
		voice->c1 = c1;
		voice->c2 = c2;
		voice->c3 = c3;

		voice->head = HEAD_PLAY_FLAG;
		voice->stream = 1;
		pthread_cond_signal(&(voice->condition));
	}
	else
	{
		ERROR_POST("Can't allocate a polyphony voice to note %i (%i voices busy)", (int)note, VOICE_COUNT);
	}
}

void* vae_sampler_new()
{
	vae_sampler* x = (vae_sampler*)pd_new(vae_sampler_class);
	if(!x)
	{
		return(0);
	}

	x->out = outlet_new(&x->obj, &s_signal);
	x->model = VaeModelCreate();
	DEBUG_POST("Create new vae_sampler~ instance");
	DEBUG_POST("CUDA is %s available", (VaeModelHasCuda(x->model) != 0) ? "" : "not");

	x->nextVoice = 0;
	memset(x->spectrograms, 0, VOICE_COUNT * MODEL_SPECTROGRAM_SIZE * sizeof(float));
	memset(x->buffers, 0, VOICE_COUNT * SAMPLE_BUFFER_SIZE * sizeof(float));
	memset(x->voices, 0, VOICE_COUNT * sizeof(poly_voice));
	memset(x->workers, 0, VOICE_COUNT * sizeof(pthread_t));
	memset(x->workerObjects, 0, VOICE_COUNT * sizeof(worker_object));

	for(int i=0; i<VOICE_COUNT; i++)
	{
		x->workerObjects[i].voiceIndex = i;
		x->workerObjects[i].sampler = x;

		pthread_cond_init(&(x->voices[i].condition), 0);
		pthread_create(&(x->workers[i]), 0, StreamVoiceSamples, &(x->workerObjects[i]));
	}

	return((void*)x);
}

void vae_sampler_free(vae_sampler* x)
{
	VaeModelDestroy(x->model);
	outlet_free(x->out);

	for(int i=0; i<VOICE_COUNT; i++)
	{
		pthread_cancel(x->workers[i]);
		pthread_cond_destroy(&(x->voices[i].condition));
	}
}

void vae_sampler_tilde_setup(void)
{
	vae_sampler_class = class_new(gensym("vae_sampler~"),
				      (t_newmethod)vae_sampler_new,
				      (t_method)vae_sampler_free,
				      sizeof(vae_sampler),
				      CLASS_DEFAULT, A_DEFFLOAT,
				      0);

	class_addmethod(vae_sampler_class, (t_method)vae_sampler_dsp, gensym("dsp"), A_NULL);
	class_addmethod(vae_sampler_class, (t_method)vae_sampler_load, gensym("load"), A_SYMBOL, A_NULL);
	class_addmethod(vae_sampler_class, (t_method)vae_sampler_fire, gensym("play"), A_FLOAT, A_FLOAT, A_FLOAT, A_FLOAT, A_FLOAT, A_NULL);

	Hann(MODEL_FFT_SIZE, HANN_WINDOW);
}
