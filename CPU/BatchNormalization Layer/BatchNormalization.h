#pragma once

#include <string>
#include <fstream>

class BatchNormalization {
private:
	void WeightLoad(CpuMat* weightName, std::string& filename);

public:
	BatchNormalization(int resultRows, int resultCols, bool useBias = true);

	void Load(std::string& betaFilename, std::string& gammaFilename, std::string& movMeanFilename, std::string& movVarFilename);
	void Apply(CpuMat* resultMat);

	float epsilon = 0.001F;
	CpuMat beta;
	CpuMat gamma;
	CpuMat movingMean;
	CpuMat movingVariance;
};
