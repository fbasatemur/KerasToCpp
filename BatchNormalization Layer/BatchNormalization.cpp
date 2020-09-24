#include <fstream>
#include "KernelBatchNormalization.cuh"
#include "BatchNormalization.h"


void BatchNormalization::weightLoad(CpuGpuMat* weightName, std::string& weightFilename)
{
	float* cpuFloatP = (float*)weightName->CpuP;
	std::ifstream file;
	file.open(weightFilename);
	std::string value;

	for (size_t i = 0; i < weightName->Size; i++) {
		file >> value;
		cpuFloatP[i] = std::stof(value);
	}

	file.close();
}

BatchNormalization::BatchNormalization(int resultRows, int resultCols, bool useBias)
{
	beta.Rows = resultRows;
	beta.Cols = useBias ? resultCols - 1 : resultCols;
	beta.Size = beta.Rows * beta.Cols;

	gamma.Rows = resultRows;
	gamma.Cols = useBias ? resultCols - 1 : resultCols;
	gamma.Size = gamma.Rows * gamma.Cols;

	movingMean.Rows = resultRows;
	movingMean.Cols = useBias ? resultCols - 1 : resultCols;
	movingMean.Size = movingMean.Rows * movingMean.Cols;

	movingVariance.Rows = resultRows;
	movingVariance.Cols = useBias ? resultCols - 1 : resultCols;
	movingVariance.Size = movingVariance.Rows * movingVariance.Cols;


	beta.cpuGpuAlloc();
	gamma.cpuGpuAlloc();
	movingMean.cpuGpuAlloc();
	movingVariance.cpuGpuAlloc();
}

void BatchNormalization::load(std::string& betaFilename, std::string& gammaFilename, std::string& movMeanFilename, std::string& movVarFilename)
{
	weightLoad(&beta, betaFilename);
	weightLoad(&gamma, gammaFilename);
	weightLoad(&movingMean, movMeanFilename);
	weightLoad(&movingVariance, movVarFilename);
}

void BatchNormalization::apply(CpuGpuMat* resultMat) {

	gpuBatchNormalization(resultMat, &beta, &gamma, &movingMean, &movingVariance, epsilon);
}

void BatchNormalization::host2Device()
{
	beta.host2Device();
	gamma.host2Device();
	movingMean.host2Device();
	movingVariance.host2Device();
}
