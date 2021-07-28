#include "CpuGpuMat.h"
#include "KernelSigmoid.cuh"
#include <iostream>


int main() {

	// input vector size( 1 row, 100 cols)
	CpuGpuMat inputImage(1, 100);

	// set values to input vector
	float* inputP = (float*)inputImage.CpuP;
	for (int i = 0; i < inputImage.Size - 1; i++)		// last value is bias = 1.0 value
		inputP[i] = (float)i;


	// copy to graphic card memory from ram
	inputImage.host2Device();


	// apply Sigmoid activation
	gpuSigmoid(&inputImage);


	// copy to ram from graphic card memory
	inputImage.device2Host();


	// show result
	float* resultP = (float*)inputImage.CpuP;
	for (int i = 0; i < inputImage.Size; i++)			// last value is bias = 1.0 value
		std::cout << resultP[i] << std::endl;

	return 0;
}