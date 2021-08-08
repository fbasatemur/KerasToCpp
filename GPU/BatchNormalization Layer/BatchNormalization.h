#pragma once
#include <string>
#include <fstream>
#include "CpuGpuMat.h"

class BatchNormalization {
private:
	void WeightLoad(CpuGpuMat* weightName, std::string& weightFilename);

public:
	BatchNormalization(int resultRows, int resultCols, bool useBias = true);
	BatchNormalization(CpuGpuMat& result, bool isEndLayer = false, bool isMemPin = false);

	void Load(std::string& betaFilename, std::string& gammaFilename, std::string& movMeanFilename, std::string& movVarFilename);
	void Apply(CpuGpuMat* resultMat);
	void Host2Device();

	float epsilon = 0.001F;
	CpuGpuMat beta;
	CpuGpuMat gamma;
	CpuGpuMat movingMean;
	CpuGpuMat movingVariance;
};
