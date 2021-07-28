#include "CpuGpuMat.h"
#include "Dense.h"
#include <iostream>


int main() {

	// input vector size( 1 row, 625 cols)
	CpuGpuMat inputImage(1, 625);

	// set values to input vector
	float* inputP = (float*)inputImage.CpuP;
	for (int i = 0; i < inputImage.Size - 1; i++)
		inputP[i] = (float)i;


	// create Dense Layers
	Dense dense(100, inputImage.Rows, inputImage.Cols);
	Dense dense1(20, dense.Result.Rows, dense.Result.Cols);
	Dense dense2(1, dense1.Result.Rows, dense1.Result.Cols, false);


	// Kernel and Bias weight addresses of each layer
	std::string denseKernel = ".\\dense\\kernel.txt";
	std::string denseBias = ".\\dense\\bias.txt";
	std::string dense1Kernel = ".\\dense_1\\kernel.txt";
	std::string dense1Bias = ".\\dense_1\\bias.txt";
	std::string dense2Kernel = ".\\dense_2\\kernel.txt";
	std::string dense2Bias = ".\\dense_2\\bias.txt";


	// load kernel and bias weights to ram
	dense.load(denseKernel, denseBias);
	dense1.load(dense1Kernel, dense1Bias);
	dense2.load(dense2Kernel, dense2Bias);


	// copy to graphic card memory from ram
	dense.host2Device();
	dense1.host2Device();
	dense2.host2Device();
	inputImage.host2Device();


	// create Neural Network

	dense.apply(&inputImage);
	// You can apply activation to the result matrix
	// gpuRelu(&dense.Result);
	// You can apply batchNormalization to the result matrix
	// batchNormalization.apply(&dense.Result);

	dense1.apply(&dense.Result);
	// You can apply activation to the result matrix
	// gpuRelu(&dense1.Result);
	// You can apply batchNormalization to the result matrix
	// batchNormalization.apply(&dense1.Result);

	dense2.apply(&dense1.Result);
	// You can apply activation to the result matrix
	// gpuSigmoid(&dense2.Result);
	// You can apply batchNormalization to the result matrix
	// batchNormalization.apply(&dense2.Result);


	// copy to ram from graphic card memory
	dense2.Result.device2Host();

	float* variancePredict = (float*)dense2.Result.CpuP;
	std::cout << variancePredict[0];

}