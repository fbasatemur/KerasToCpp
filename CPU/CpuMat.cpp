#include "CpuMat.h"
#include <fstream>


CpuMat::CpuMat() {}

CpuMat::CpuMat(const size_t& rows, const size_t& cols, const size_t& depth, bool useBias) {
	this->Rows = rows;
	this->Cols = useBias ? cols + 1 : cols;
	this->Depth = depth;
	this->Size = this->Rows * this->Cols * this->Depth;

	MemAlloc();

	if (useBias) {
		float* biasValue = (float*)this->CpuP;
		biasValue[this->Size - 1] = 1.0F;
	}
}

CpuMat::~CpuMat() {

	free(this->CpuP);
}

void CpuMat::MemAlloc() {

	this->CpuP = (void*)malloc(getAllocationSize());
}

int CpuMat::getAllocationSize() {

	return this->Size * sizeof(float);
}


std::string* ReadTxtToBuffer(std::string& filename)
{
	std::ifstream f(filename);
	std::string* buffer = new std::string;
	f.seekg(0, std::ios::end);
	buffer->resize(f.tellg());
	f.seekg(0);
	f.read((char*)buffer->data(), buffer->size());
	f.close();
	return buffer;
}