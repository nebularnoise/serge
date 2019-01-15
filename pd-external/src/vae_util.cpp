//*****************************************************************
//
//	$file: vae_util.cpp $
//	$date: 02/01/2019 $
//	$revision: $
//
//*****************************************************************
#include<torch/script.h>
#include<torch/csrc/api/include/torch/cuda.h>
#include<assert.h>
#include"profile.h"

//-----------------------------------------------------------------
// vae_model wrapper struct
//-----------------------------------------------------------------

typedef struct vae_model_t
{
	bool hasCuda;
	std::shared_ptr<torch::jit::script::Module> module;
}vae_model;

//-----------------------------------------------------------------
// vae_model wrapper functions
//-----------------------------------------------------------------

extern "C" int VaeModelHasCuda(vae_model* model)
{
	return(model ? (model->hasCuda ? 1 : 0) : 0);
}

extern "C" vae_model* VaeModelCreate()
{
	vae_model* model = new vae_model;
	model->module = 0;
	model->hasCuda = torch::cuda::is_available();
	return(model);
}

extern "C" void VaeModelDestroy(vae_model* model)
{
	model->module.reset();
	delete model;
}

extern "C" int VaeModelLoad(vae_model* model, const char* path)
{
	model->module.reset();
	try
	{
		model->module = torch::jit::load(path);
		if(model->hasCuda)
		{
			model->module->to(at::kCUDA);
		}
	}
	catch(...)
	{
		return(-1);
	}
	return(0);
}


#define clamp(x, low, hi) ((x) > (hi)) ? (hi) : (((x)<(low))? (low) : (x))

extern "C" int VaeModelGetSpectrogram(vae_model* model, unsigned int count, float* buffer, float c0, float c1, float c2, float c3, int note)
{
	try
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

			torch::Tensor coordsTensor = torch::from_blob(coordsArray, {1, 4}).to(model->hasCuda ? at::kCUDA : at::kCPU);
			torch::Tensor octaveTensor = torch::from_blob(octaveSelector, {1, 7}).to(model->hasCuda ? at::kCUDA : at::kCPU);
			torch::Tensor pitchTensor = torch::from_blob(pitchSelector, {1, 12}).to(model->hasCuda ? at::kCUDA : at::kCPU);

			std::vector<at::IValue> v = {coordsTensor, octaveTensor, pitchTensor};

			torch::Tensor out;

			TIME_BLOCK_START();
			out = model->module->forward(v).toTensor().to(at::kCPU);
			TIME_BLOCK_END("Torch forward");

			auto a = out.accessor<float, 2>();
			if(count != a.size(0)*a.size(1))
			{
				return(-2);
			}

			int index = 0;

			TIME_BLOCK_START();
			for(int bin=0; bin<a.size(0); bin++)
			{
				for(int slice = 0; slice<a.size(1); slice++)
				{
					buffer[slice * a.size(0) + bin] = a[bin][slice];
					index++;
					if(index > count)
					{
						return(-2);
					}
				}
			}
			TIME_BLOCK_END("Copy spectrogram tensor");
			return(0);
		}
		else
		{
			return(-1);
		}
	}
	catch(...)
	{
		assert(0);
	}
}
