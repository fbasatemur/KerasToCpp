#include "CpuGpuMat.h"
#include "Dense.h"
#include <iostream>


int main() {

	// create Dense Layers
	Dense dense(100, 1, 100);
	Dense dense1(20, dense.Result);
	Dense dense2(1, dense1.Result, true);


	// Kernel and Bias weight addresses of each layer
	std::string denseKernel = ".\\dense\\kernel.txt";
	std::string denseBias = ".\\dense\\bias.txt";
	std::string dense1Kernel = ".\\dense_1\\kernel.txt";
	std::string dense1Bias = ".\\dense_1\\bias.txt";
	std::string dense2Kernel = ".\\dense_2\\kernel.txt";
	std::string dense2Bias = ".\\dense_2\\bias.txt";


	// load kernel and bias weights to ram
	dense.Load(denseKernel, denseBias);
	dense1.Load(dense1Kernel, dense1Bias);
	dense2.Load(dense2Kernel, dense2Bias);


	// input vector size( 1 row, 100 cols)
	CpuGpuMat inputImage(1, 100);
	CpuGpuMat output(dense2.Result, 1);

	// set values to input vector
	float* inputP = (float*)inputImage.CpuP;
	for (int i = 0; i < inputImage.Size - 1; i++)
		inputP[i] = (float)i;


	// copy to graphic card memory from ram
	dense.Host2Device();
	dense1.Host2Device();
	dense2.Host2Device();
	inputImage.Host2Device();


	// create Neural Network

	dense.Apply(&inputImage);
	// You can apply activation to the result matrix
	// gpuRelu(&dense.Result);
	// You can apply batchNormalization to the result matrix
	// batchNormalization.apply(&dense.Result);

	dense1.Apply(&dense.Result);
	// You can apply activation to the result matrix
	// gpuRelu(&dense1.Result);
	// You can apply batchNormalization to the result matrix
	// batchNormalization.apply(&dense1.Result);

	dense2.Apply(&dense1.Result);
	// You can apply activation to the result matrix
	// gpuSigmoid(&dense2.Result);
	// You can apply batchNormalization to the result matrix
	// batchNormalization.apply(&dense2.Result);


	// copy to ram from graphic card memory
	output.Device2Host();

	float* variancePredict = (float*)output.CpuP;
	std::cout << variancePredict[0];

}
