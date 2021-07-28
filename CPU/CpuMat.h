#pragma once
#include <string>

class CpuMat {

public:
	void* CpuP;		// ram pointer
	size_t Rows;
	size_t Cols;
	size_t Depth;
	size_t Size;		// rows * cols * depth

	CpuMat();
	CpuMat(const size_t& rows, const size_t& cols, const size_t& depth = 1, bool useBias = true);
	~CpuMat();				// memory free

	void MemAlloc();		// memory allocation

private:
	int getAllocationSize();
};

std::string* ReadTxtToBuffer(std::string& filename);