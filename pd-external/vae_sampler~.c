//*****************************************************************
//
//	$file: vae_sampler~.c $
//	$date: 28/10/2018 $
//	$revision: $
//
//*****************************************************************
#include<stdlib.h>
#include"m_pd.h"
#include"vae_util.h"

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

const int SAMPLE_BUFFER_SIZE = 34560;

typedef struct vae_sampler_t
{
	t_object	obj;
	t_outlet*	out;

	vae_model*	model;
	int		play;
	int		counter;
	float		buffer[SAMPLE_BUFFER_SIZE];

} vae_sampler;

static	t_class* vae_sampler_class;
//-----------------------------------------------------------------
// methods
//-----------------------------------------------------------------

t_int* vae_sampler_perform(t_int* w)
{
	vae_sampler* x		= (vae_sampler*)w[1];
	t_sample* out	= (t_sample*)w[2];
	int n		= (int)w[3];

	int i=0;
	if(x->play)
	{
		for(i=0; i<n && x->counter<SAMPLE_BUFFER_SIZE; i++)
		{
			out[i] = x->buffer[x->counter];
			x->counter++;
		}
	}
	for(; i<n; i++)
	{
		out[i] = 0;
	}
	if(x->counter >= SAMPLE_BUFFER_SIZE)
	{
		x->counter = 0;
		x->play = 0;
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

void vae_sampler_fire(vae_sampler* x, t_symbol* sym, float c0, float c1, float c2, float c3, float nu)
{
	DEBUG_POST("Fire : %f %f %f %f", c1, c2, c3, nu);

	if(VaeModelGetSamples(x->model, SAMPLE_BUFFER_SIZE, x->buffer, c0, c1, c2, c3, nu))
	{
		ERROR_POST("Failed to get samples from model...");
	}
	else
	{
		DEBUG_POST("Got samples from model");
		x->counter = 0;
		x->play = 1;
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
	x->play = 0;
	x->counter = 0;
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
}
