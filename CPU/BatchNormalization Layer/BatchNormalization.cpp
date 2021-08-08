#include <fstream>
#include "KernelCpu.h"
#include "BatchNormalization.h"


void BatchNormalization::Weightload(CpuMat* weightName, std::string& filename)
{
	float* cpuFloatP = (float*)weightName->CpuP;

	std::string* buffer = ReadTxtToBuffer(filename);
	char* pch = strtok((char*)buffer->c_str(), " \n");

	for (size_t i = 0; i < weightName->Size; i++)
	{
		cpuFloatP[i] = std::stof(pch);
		pch = strtok(NULL, " \n");
	}

	delete buffer;
	delete[] pch;
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


	beta.MemAlloc();
	gamma.MemAlloc();
	movingMean.MemAlloc();
	movingVariance.MemAlloc();
}

void BatchNormalization::Load(std::string& betaFilename, std::string& gammaFilename, std::string& movMeanFilename, std::string& movVarFilename)
{
	WeightLoad(&beta, betaFilename);
	WeightLoad(&gamma, gammaFilename);
	WeightLoad(&movingMean, movMeanFilename);
	WeightLoad(&movingVariance, movVarFilename);
}


void BatchNormalization::Apply(CpuMat* resultMat) {

	cpuBatchNormalization(resultMat, &beta, &gamma, &movingMean, &movingVariance, epsilon);
}
