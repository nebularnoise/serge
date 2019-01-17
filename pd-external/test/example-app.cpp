#include <torch/torch.h>
#include <iostream>

int main()
{
	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << std::endl;

	try
	{
		if(at::hasCUDA() && torch::cuda::is_available())
		{
		      std::cout << "CUDA is available" << std::endl;
		}
		else
		{
		      std::cout << "CUDA is not available" << std::endl;
		}
	}
	catch(...)
	{
		std::cout << "CUDA driver not installed" << std::endl;
	}
}
