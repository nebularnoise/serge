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
#include"m_pd.h"
#include"vae_util.h"
#include"griffin_lim.h"

//-----------------------------------------------------------------
// debug printing macros
//-----------------------------------------------------------------

#ifdef DEBUG
	#define DEBUG_POST(s, ...) post(s, ##__VA_ARGS__)
#else
	#define DEBUG_POST(s, ...)
#endif
#define ERROR_POST(s, ...) error(s, ##__VA_ARGS__)

//-----------------------------------------------------------------
// object definition
//-----------------------------------------------------------------

#define GRIFFIN_LIM_ITERATION_COUNT 50

#define MODEL_SLICE_COUNT	128
#define MODEL_FFT_SIZE		2048
#define MODEL_BIN_COUNT		1025
#define MODEL_SPECTROGRAM_SIZE	MODEL_SLICE_COUNT * MODEL_BIN_COUNT
#define MODEL_HOP_SIZE		MODEL_FFT_SIZE / 8
#define MODEL_OLA_GAIN		3
#define SAMPLE_BUFFER_SIZE	MODEL_SLICE_COUNT * MODEL_HOP_SIZE + MODEL_FFT_SIZE
#define	VOICE_COUNT		16

typedef int voice_head;
const voice_head HEAD_PLAY_FLAG    = 1<<31,
                 HEAD_COUNTER_MASK = ~(1<<31);

static float HANN_WINDOW[MODEL_FFT_SIZE];

typedef struct poly_voice_t
{
	float note;
	voice_head head;

} poly_voice;

typedef struct vae_sampler_t
{
	t_object	obj;
	t_outlet*	out;

	vae_model*	model;

	int		nextVoice;
	float		spectrogram[MODEL_SPECTROGRAM_SIZE];
	float		buffers[VOICE_COUNT][SAMPLE_BUFFER_SIZE];
	poly_voice	voices[VOICE_COUNT];

} vae_sampler;

static	t_class* vae_sampler_class;
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

		if(head & HEAD_PLAY_FLAG)
		{
			int counter = head & HEAD_COUNTER_MASK;

			for(int i=0; i<n && counter<SAMPLE_BUFFER_SIZE; i++)
			{
				out[i] += buffer[counter];
				counter++;
			}
			if(counter >= SAMPLE_BUFFER_SIZE)
			{
				voice->head = 0;
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
		DEBUG_POST("Loaded model %s", sym->s_name);
	}
}

void vae_sampler_fire(vae_sampler* x, t_symbol* sym, float c0, float c1, float c2, float c3, float note)
{
	for(int i=0; i<VOICE_COUNT; i++)
	{
		if(x->voices[i].note == note)
		{
			//NOTE: the given note is already playing... (we don't retrigger, but maybe we should ?)
			return;
		}
	}

	DEBUG_POST("Fire : %f %f %f %f, note %i", c0, c1, c2, c3, (int)floorf(note));

	int voice = x->nextVoice;
	float* spectrogram = x->spectrogram;
	int err = 0;
	if((err = VaeModelGetSpectrogram(x->model, MODEL_SPECTROGRAM_SIZE, spectrogram, c0, c1, c2, c3, (int)floorf(note))))
	{
		ERROR_POST("Failed to get spectrogram from model (%s)...", (err == -1) ? "no module" : "wrong tensor dimensions");
	}
	else
	{
		DEBUG_POST("Got spectrogram from model");

		GriffinLimReconstruct(GRIFFIN_LIM_ITERATION_COUNT,
				      MODEL_FFT_SIZE,
				      MODEL_HOP_SIZE,
				      MODEL_SLICE_COUNT,
				      HANN_WINDOW,
				      MODEL_OLA_GAIN,
				      spectrogram,
				      x->buffers[voice]);

		x->voices[voice].head = HEAD_PLAY_FLAG;
		x->voices[voice].note = note;
		x->nextVoice++;
		if(x->nextVoice >= VOICE_COUNT)
		{
			x->nextVoice = 0;
		}
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

	memset(x->buffers, 0, VOICE_COUNT*SAMPLE_BUFFER_SIZE*sizeof(float));
	memset(x->voices, 0, VOICE_COUNT*sizeof(poly_voice));

	return((void*)x);
}

void vae_sampler_free(vae_sampler* x)
{
	VaeModelDestroy(x->model);
	outlet_free(x->out);
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
