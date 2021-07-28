#pragma once
#include <string>
#include <fstream>
#include "CpuGpuMat.h"

class BatchNormalization {
private:
	void weightLoad(CpuGpuMat* weightName, std::string& weightFilename);

public:
	BatchNormalization(int resultRows, int resultCols, bool useBias = true);

	void load(std::string& betaFilename, std::string& gammaFilename, std::string& movMeanFilename, std::string& movVarFilename);
	void apply(CpuGpuMat* resultMat);
	void host2Device();

	float epsilon = 0.001F;
	CpuGpuMat beta;
	CpuGpuMat gamma;
	CpuGpuMat movingMean;
	CpuGpuMat movingVariance;
};