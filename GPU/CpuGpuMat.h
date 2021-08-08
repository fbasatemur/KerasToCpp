#pragma once

class CpuGpuMat {

public:
	void* CpuP;		// ram pointer
	void* GpuP;		// graphic memory pointer
	int Rows;
	int Cols;
	int Size;		// rows * cols
	bool MemPinned;
	void* deallocCpuP = nullptr;		// for free memory ram pointer
	void* deallocGpuP = nullptr;		// for free memory gpu pointer

	CpuGpuMat();
	CpuGpuMat(int rows, int cols, bool useMemPin = false);
	CpuGpuMat(const float* inputs, int inputRow, int inputCol, int numberInputs, bool useMemPin = false);
	CpuGpuMat(CpuGpuMat& result, int numberOutputs, bool useMemPin = false);
	~CpuGpuMat();			// memory free
	void Host2Device();
	void Device2Host();
	void CpuGpuAlloc();		// memory allocation

private:
	int GetAllocationSize();
};

