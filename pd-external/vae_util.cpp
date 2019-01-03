//*****************************************************************
//
//	$file: vae_util.cpp $
//	$date: 02/01/2019 $
//	$revision: $
//	$note: (C) 2019 by Martin Fouilleul - all rights reserved $
//
//*****************************************************************
#include<torch/script.h>

//-----------------------------------------------------------------
// vae_model wrapper struct
//-----------------------------------------------------------------

typedef struct vae_model_t
{
	std::shared_ptr<torch::jit::script::Module> module;
}vae_model;

//-----------------------------------------------------------------
// vae_model wrapper functions
//-----------------------------------------------------------------

extern "C" vae_model* VaeModelCreate()
{
	vae_model* model = new vae_model;
	model->module = 0;
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
	}
	catch(...)
	{
		return(-1);
	}
	return(0);
}

#include<stdio.h>

extern "C" int VaeModelGetSamples(vae_model* model, unsigned int count, float* buffer, float c0, float c1, float c2, float c3, float nu)
{
	if(model->module)
	{
		float inputs[5] = {c0, c1, c2, c3, nu};
		torch::Tensor in = torch::from_blob(inputs, {1, 5});

		std::vector<at::IValue> v = {in};
		torch::Tensor out = model->module->forward(v).toTensor();

		auto a = out.accessor<float, 2>();
		for(int i=0; i<a.size(1) && i<count;i++)
		{
			buffer[i] = a[0][i];
		}
		return(0);
	}
	else
	{
		return(-1);
	}
}
