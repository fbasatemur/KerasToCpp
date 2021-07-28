#pragma once

#include <string>
#include <fstream>

class BatchNormalization {
private:
	void weightLoad(CpuMat* weightName, std::string& filename);

public:
	BatchNormalization(int resultRows, int resultCols, bool useBias = true);

	void load(std::string& betaFilename, std::string& gammaFilename, std::string& movMeanFilename, std::string& movVarFilename);
	void apply(CpuMat* resultMat);

	float epsilon = 0.001F;
	CpuMat beta;
	CpuMat gamma;
	CpuMat movingMean;
	CpuMat movingVariance;
};