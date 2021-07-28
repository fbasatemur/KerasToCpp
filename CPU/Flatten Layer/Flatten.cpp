#include "Flatten.h"

Flatten::Flatten(CpuMat* input, bool useBias)
{
	Result = new CpuMat(1, input->Size, 1, useBias);
}

void Flatten::apply(CpuMat* input)
{
	float* resultP = (float*)Result->CpuP;
	float* inputP = (float*)input->CpuP;

	size_t input2DSize = input->Rows * input->Cols;
	size_t resultIndex = 0;

	for (size_t r = 0; r < input->Rows; r++)
		for (size_t c = 0; c < input->Cols; c++)
			for (size_t d = 0; d < input->Depth; d++) {
				resultP[resultIndex] = inputP[d * input2DSize + r * input->Cols + c];
				resultIndex++;
			}
}