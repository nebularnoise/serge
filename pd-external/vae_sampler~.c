//*****************************************************************
//
//	$file: vae_sampler~.c $
//	$date: 28/10/2018 $
//	$revision: $
//
//*****************************************************************
#include<stdlib.h>
#include"m_pd.h"

//-----------------------------------------------------------------
// util
//-----------------------------------------------------------------

#ifdef DEBUG
#define DEBUG_POST(s, ...) post(s, ##__VA_ARGS__)
#else
#define DEBUG_POST(s, ...)
#endif

//-----------------------------------------------------------------
// object definition
//-----------------------------------------------------------------

typedef struct vae_sampler_t
{
	t_object	obj;
	t_outlet*	out;

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

	for(int i=0; i<n; i++)
	{
		out[i] = 0;
	}
	return(w+4);
}

void vae_sampler_dsp(vae_sampler* x, t_signal** sp)
{
	dsp_add(vae_sampler_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

void vae_sampler_load(vae_sampler* x, t_symbol* sym)
{
	DEBUG_POST("Load received : %s", sym->s_name);

	//TODO: implement pytorch model import
}

void vae_sampler_fire(vae_sampler* x, t_symbol* sym, float c1, float c2, float c3, float c4)
{
	DEBUG_POST("Fire : %f %f %f %f", c1, c2, c3, c4);

	//TODO: fill the output buffer with the model's output
}

void* vae_sampler_new()
{
	vae_sampler* x = (vae_sampler*)pd_new(vae_sampler_class);
	if(!x)
	{
		return(0);
	}

	x->out = outlet_new(&x->obj, &s_signal);

	return((void*)x);
}

void vae_sampler_free(vae_sampler* x)
{
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
	class_addmethod(vae_sampler_class, (t_method)vae_sampler_fire, gensym("play"), A_FLOAT, A_FLOAT, A_FLOAT, A_FLOAT, A_NULL);
}
