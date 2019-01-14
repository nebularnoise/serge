//*****************************************************************
//
//	$file: vae_util.h $
//	$date: 02/01/2019 $
//	$revision: $
//
//*****************************************************************
#ifndef __VAE_UTIL_H_
#define __VAE_UTIL_H_

//-----------------------------------------------------------------
// vae_model wrapper
//-----------------------------------------------------------------

typedef struct vae_model_t vae_model;

vae_model* VaeModelCreate();
void VaeModelDestroy(vae_model* model);
int  VaeModelLoad(vae_model* model, const char* path);
int VaeModelGetSpectrogram(vae_model* model, unsigned int count, float* buffer, float c0, float c1, float c2, float c3, int note);

#endif //__VAE_UTIL_H_
