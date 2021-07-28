#include "CpuMat.h"
#include "KernelCpu.h"
#include <math.h>

void cpuConv2D(float* input, int inputRows, int inputCols, int inputDepth, int filters, CpuMat* kernel, CpuMat* result)
{
	float sum;
	float* kernelP, *resultP;
	size_t kernel2DSize = kernel->Cols * kernel->Rows;
	size_t input2DSize = inputRows * inputCols;
	size_t result2DSize = result->Cols * result->Rows;
	size_t kernelBiasPos = kernel->Size - 1;

	for (size_t f = 0; f < filters; f++)
	{
		kernelP = (float*)kernel[f].CpuP;
		resultP = (float*)result->CpuP + f * result2DSize;

		for (size_t r = 0; r < result->Rows; r++)
		{
			for (size_t c = 0; c < result->Cols; c++)
			{
				sum = 0.0F;
				for (size_t row = 0; row < kernel->Rows; row++)
				{
					for (size_t col = 0; col < kernel->Cols; col++)
					{
						for (size_t depth = 0; depth < kernel->Depth; depth++)
						{
							sum += input[depth * input2DSize + (r + row)* inputRows + c + col] * kernelP[depth * kernel2DSize + row * kernel->Rows + col];
						}
					}
				}
				resultP[r * result->Rows + c] = sum + kernelP[kernelBiasPos];			// add bias value
			}
		}
	}
}

void cpuMatrixMult(float* cpuMat1, float* cpuMat2, float* cpuMat3, int m1Rows, int m1Cols, int m2Cols)
{
	float sum = 0.0;
	for (int row = 0; row < m1Rows; row++)
	{
		for (int col = 0; col < m2Cols; col++)
		{
			sum = 0.0;
			for (int i = 0; i < m1Cols; i++)
			{
				sum += cpuMat1[row * m1Cols + i] * cpuMat2[i * m2Cols + col];
			}
			cpuMat3[row * m2Cols + col] = sum;
		}
	}
}

void cpuNormAndShift(float* CpuP, int size, float mean, float variance, float epsilon, float beta, float gamma)
{
	for (int i = 0; i < size; i++)
		CpuP[i] = (CpuP[i] - mean) / (float)sqrt(variance + epsilon) * gamma + beta;
}

void cpuReluActivation(float* CpuP, int size)
{
	for (int i = 0; i < size; i++)
		CpuP[i] = CpuP[i] > 0 ? CpuP[i] : 0;
}

void cpuSigmoidActivation(float* CpuP, int size)
{
	for (int i = 0; i < size; i++)
		CpuP[i] = (float)(1.0 / (1.0 + exp(-1.0 * (double)CpuP[i])));
}

void cpuBatchNorm(float* cpuResult, float* cpuBeta, float* cpuGamma, float* cpuMovingMean, float* cpuMovingVar, float epsilon, int size)
{
	for (int i = 0; i < size; i++)
		cpuResult[i] = (cpuResult[i] - cpuMovingMean[i]) / (float)sqrt(cpuMovingVar[i] + epsilon) * cpuGamma[i] + cpuBeta[i];
}


void cpuMatrixConv2D(CpuMat* input, int filters, CpuMat* kernel, CpuMat* result)
{
	cpuConv2D((float*)input->CpuP, input->Rows, input->Cols, input->Depth, filters, kernel, result);
}

void cpuMatrixMultiplication(CpuMat* Mat1, CpuMat* Mat2, CpuMat* Mat3)
{
	cpuMatrixMult((float*)Mat1->CpuP, (float*)Mat2->CpuP, (float*)Mat3->CpuP, Mat1->Rows, Mat1->Cols, Mat2->Cols);
}

void cpuNormalizeAndShift(CpuMat* Mat, float mean, float variance, float epsilon, float beta, float gama)
{
	cpuNormAndShift((float*)Mat->CpuP, Mat->Size, mean, variance, epsilon, beta, gama);
}

void cpuRelu(CpuMat* Mat)
{
	cpuReluActivation((float*)Mat->CpuP, Mat->Size);
}

void cpuSigmoid(CpuMat* Mat)
{
	cpuSigmoidActivation((float*)Mat->CpuP, Mat->Size);
}

void cpuBatchNormalization(CpuMat* result, CpuMat* beta, CpuMat* gamma, CpuMat* movingMean, CpuMat* movingVariance, float epsilon)
{
	cpuBatchNorm((float*)result->CpuP, (float*)beta->CpuP, (float*)gamma->CpuP, (float*)movingMean->CpuP, (float*)movingVariance->CpuP, epsilon, beta->Size);
}