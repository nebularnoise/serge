//*****************************************************************
//
//	$file: vae_sampler~.c $
//	$date: 28/10/2018 $
//	$revision: $
//
//*****************************************************************
#include<stdlib.h>	// malloc
#include<string.h>	// memset
#include<math.h>	// M_PI, cos, ...
#include<pthread.h>
#include"m_pd.h"
#include"vae_util.h"
#include"griffin_lim.h"
#include"profile.h"

//-----------------------------------------------------------------
// debug printing macros
//-----------------------------------------------------------------

#ifdef DEBUG
	#define DEBUG_POST(s, ...) post(s, ##__VA_ARGS__)
#else
	#define DEBUG_POST(s, ...)
#endif
#define ERROR_POST(s, ...) error(s, ##__VA_ARGS__)
#define POST(s, ...) post(s, ##__VA_ARGS__)

//-----------------------------------------------------------------
// object definition
//-----------------------------------------------------------------

//TODO(martin): would be much safer with const int (esp. with respect to parenthesing)
//		but gcc on linux seems to fail when instancing buffers with const length...??

#define GRIFFIN_LIM_ITERATION_COUNT 60

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
	volatile int		note;
	volatile voice_head	head;
	volatile int		endCursor;

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

#include<stdio.h>
#include<assert.h>

void* StreamGriffinLim(void* x)
{
	worker_object* object = (worker_object*)x;
	vae_sampler* sampler = object->sampler;
	int voiceIndex = object->voiceIndex;

	poly_voice* voice = &(sampler->voices[voiceIndex]);
	float* spectrogram = sampler->spectrograms[voiceIndex];
	float* samplesBuffer = &(sampler->buffers[voiceIndex][0]);

	float batchBuffer[GL_BATCH_SAMPLES];
	memset(batchBuffer, 0, GL_BATCH_SAMPLES*sizeof(float));

	while(1)
	{
		while(!voice->note)
		{
			//NOTE(martin): wait for our voice to be allocated
		}
		DEBUG_POST("Wake up worker thread %i for note %i", voiceIndex, voice->note);

		//NOTE(martin): stream griffin lim batches

		memset(samplesBuffer, 0, SAMPLE_BUFFER_SIZE*sizeof(float));

		int batchStart = 0;
		int samplesStart = 0;

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
		voice->endCursor = SAMPLE_BUFFER_SIZE+1;
		DEBUG_POST("Worker thread %i finished decoding note %i", voiceIndex, voice->note);
		while(voice->head)
		{
			//NOTE(martin): wait for our sampler to finish reading the buffer
		}
		DEBUG_POST("Worker thread %i go to sleep", voiceIndex);
		voice->endCursor = 0;
		voice->note = 0;
	}
	return(0);
}

//-----------------------------------------------------------------
// methods
//-----------------------------------------------------------------

void Hann(int count, float* windowOut)
{
	float invCount = 1./(count-1);
	for(int i=0; i<count; i++)
	{
		windowOut[i] = 0.5*(1 - cos(2*M_PI*i*invCount));
	}
}

t_int* vae_sampler_perform(t_int* w)
{
	vae_sampler* x		= (vae_sampler*)w[1];
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
				DEBUG_POST("Perform routine finished playing voice %i", voiceIndex);
				voice->head = 0;
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

void vae_sampler_fire(vae_sampler* x, t_symbol* sym, float c0, float c1, float c2, float c3, float fnote)
{
	int note = (int)floorf(fnote);
	for(int i=0; i<VOICE_COUNT; i++)
	{
		if(x->voices[i].note == note)
		{
			//NOTE: the given note is already playing... (we don't retrigger, but maybe we should ?)
			DEBUG_POST("vae_sampler_fire(): note %i is already playing");
			return;
		}
	}

	DEBUG_POST("Play (%f %f %f %f), note %i", c0, c1, c2, c3, note);

	int voiceIndex = x->nextVoice;
	poly_voice* voice = &(x->voices[voiceIndex]);

	if(!voice->note)
	{
		float* spectrogram = x->spectrograms[voiceIndex];
		int err = 0;

		TIME_BLOCK_START();
		err = VaeModelGetSpectrogram(x->model, MODEL_SPECTROGRAM_SIZE, spectrogram, c0, c1, c2, c3, note);
		TIME_BLOCK_END("VaeModelGetSpectrogram()");

		if(err)
		{
			ERROR_POST("Failed to get spectrogram from model (%s)...", (err == -1) ? "no module" : "wrong tensor dimensions");
		}
		else
		{
			DEBUG_POST("Got spectrogram from model");

			x->nextVoice++;
			if(x->nextVoice >= VOICE_COUNT)
			{
				x->nextVoice = 0;
			}
			voice->endCursor = 0;
			voice->note = note;
			voice->head = HEAD_PLAY_FLAG;
		}
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
		pthread_create(&(x->workers[i]), 0, StreamGriffinLim, &(x->workerObjects[i]));
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
