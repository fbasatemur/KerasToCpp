#include <iostream>
#include "CpuGpuMat.h"
#include "BatchNormalization.h"

int main() {

	// create input matrix
	CpuGpuMat inputImage(1, 100);

	// set values to input matrix
	float* inputP = (float*)inputImage.CpuP;
	for (int i = 0; i < inputImage.Size - 1; i++)		// last value is bias = 1.0 value
		inputP[i] = (float)i;


	// gamma = 1.0F, beta = 0.0F, epsilon = 0.001F -> Keras Default Hiperparameters
	// create Keras BatchNormalization Layer
	BatchNormalization batchNorm(inputImage.Rows, inputImage.Cols);


	// load beta, gamma, moving_mean and moving_variance weights to ram
	std::string batchNormBeta = ".\\batch_normalization\\beta.txt";
	std::string batchNormGamma = ".\\batch_normalization\\gamma.txt";
	std::string batchNormMovingMean = ".\\batch_normalization\\moving_mean.txt";
	std::string batchNormMovingVariance = ".\\batch_normalization\\moving_variance.txt";

	// load batchnormalization layer weights to ram
	batchNorm.Load(batchNormBeta, batchNormGamma, batchNormMovingMean, batchNormMovingVariance);


	// copy to graphic card memory from ram
	batchNorm.Host2Device();
	inputImage.Host2Device();


	// Apply batchNormalization to inputImage matrix
	batchNorm.Apply(&inputImage);


	// copy to ram from graphic card memory
	inputImage.Device2Host();


	// show result
	float* resultP = (float*)inputImage.CpuP;
	for (int i = 0; i < inputImage.Size; i++)			// last value is bias = 1.0 value
		std::cout<< resultP[i] << std::endl;

	return 0;
}
