/*************************************************************//**
*
*	@file	vae_util.cpp
*	@date	02/01/2019
*	@author Martin Fouilleul
*
*****************************************************************/
#include<torch/script.h>
#include<torch/csrc/api/include/torch/cuda.h>
#include<assert.h>
#include"profile.h"
#include"vae_util.h"

//-----------------------------------------------------------------
// vae_model wrapper struct
//-----------------------------------------------------------------

typedef struct vae_model_t
{
	bool hasCuda;
	torch::Device device;
	std::shared_ptr<torch::jit::script::Module> module;

	vae_model_t():device(torch::kCPU){}
}vae_model;

//-----------------------------------------------------------------
// vae_model wrapper functions
//-----------------------------------------------------------------

/**
	@brief Test if CUDA is available

	@param model The VAE model to test
	@return 1 if the VAE model has CUDA capability, 0 otherwise
*/
extern "C" int VaeModelHasCuda(vae_model* model)
{
	if(!model)
	{
		return(0);
	}
	else
	{
		return(model->hasCuda ? 1 : 0);
	}
}

/**
	@brief Create a VAE model
	@return The newly created model.
*/
extern "C" vae_model* VaeModelCreate()
{
	vae_model* model = new vae_model;
	model->module = 0;

	#ifndef USE_CUDA
		#warning "CUDA support is disabled by default. You can add CUDA support by defining CUDA=true"
		model->hasCuda = false;
	#else
		try
		{
			model->hasCuda = (torch::cuda::device_count() != 0);
			if(model->hasCuda)
			{
				model->device = torch::Device(torch::kCUDA);
			}
			else
			{
				model->device = torch::Device(torch::kCPU);
			}
		}
		catch(...)
		{
			model->hasCuda = false;
		}
	#endif

	return(model);
}

/**
	@brief Destroys a VAE model
	@param model The model to destroy
*/
extern "C" void VaeModelDestroy(vae_model* model)
{
	model->module.reset();
	delete model;
}

/**
	@brief Load a module from a serialized torchscript file
	@param path Path to the serialized torchscript module
	@return 0 if the module was loaded, -1 otherwise
*/
extern "C" int VaeModelLoad(vae_model* model, const char* path)
{
	model->module.reset();
	try
	{
		model->module = torch::jit::load(path);
		model->module->to(model->device);
	}
	catch(const std::exception& e)
	{
		printf("%s\n", e.what());
		return(-1);
	}
	return(0);
}


#define clamp(x, low, hi) ((x) > (hi)) ? (hi) : (((x)<(low))? (low) : (x))

/**
	@brief Get a spectrogram from a VAE model

	@param count The size of the output buffer
	@param buffer An output buffer which will hold the spectrogram
	@param c0	First coordinate in latent space
	@param c1	Second coordinate in latent space
	@param c2	Third coordinate in latent space
	@param c3	Fourth coordinate in latent space
	@param note	Requested MIDI note

	@return VAE_MODEL_OK if no error occured.
*/
extern "C" int VaeModelGetSpectrogram(vae_model* model, unsigned int count, float* buffer, float c0, float c1, float c2, float c3, int note)
{
	if(model->module)
	{
		//NOTE(martin): middle C (midi note 60) is C4, so A0 is 21

		float octaveSelector[7] = {0};
		float pitchSelector[12] = {0};

		int octave = (note - 21) / 12 ;
		int pitchClass = (note - 21) - octave*12;

		//TODO(martin): error / assert rather than clamp ?

		octave = clamp(octave, 0, 6);
		pitchClass = clamp(pitchClass, 0, 11);

		octaveSelector[octave] = 1;
		pitchSelector[pitchClass] = 1;

		float coordsArray[4] = {c0, c1, c2, c3};

		//NOTE(martin): the model takes 3 tensor as input
		//		first tensor of dimension (1x4) is the latent space coordinates
		//		second tensor of dimension (1x7) is an octave one-hot
		//		third tensor of dimension (1x12) is a pitch class one-hot

		torch::Tensor coordsTensor = torch::from_blob(coordsArray, {1, 4}).to(model->device);
		torch::Tensor octaveTensor = torch::from_blob(octaveSelector, {1, 7}).to(model->device);
		torch::Tensor pitchTensor = torch::from_blob(pitchSelector, {1, 12}).to(model->device);

		std::vector<at::IValue> v = {coordsTensor, octaveTensor, pitchTensor};

		torch::Tensor out;

		try
		{
			TIME_BLOCK_START();
			out = model->module->forward(v).toTensor().to(torch::Device(torch::kCPU));
			TIME_BLOCK_END("Torch forward");
		}
		catch(const std::exception& e)
		{
			printf("%s\n", e.what());
			return(VAE_MODEL_THROW);
		}

		auto a = out.accessor<float, 2>();
		if(count != a.size(0)*a.size(1))
		{
			return(VAE_MODEL_BAD_SIZE);
		}

		TIME_BLOCK_START();
		for(int bin=0; bin<a.size(0); bin++)
		{
			for(int slice = 0; slice<a.size(1); slice++)
			{
				buffer[slice * a.size(0) + bin] = a[bin][slice];
			}
		}
		TIME_BLOCK_END("Copy spectrogram tensor");
		return(VAE_MODEL_OK);
	}
	else
	{
		return(VAE_MODEL_NOT_LOADED);
	}
}
