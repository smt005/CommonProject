#include "Test.h"

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <string>
#include <thread>
#include <stdio.h>

namespace {
	void fun0Cpu(int* count, int* arrayInt, int* arrayIntResult) {
		int val = 0;

		for (int i = 0; i < *count; ++i) {
			arrayIntResult[i] = arrayInt[i] * arrayInt[i];
		}
	}

	void fun1Cpu(int* count, int* arrayInt, int* sum) {
		*sum = 0;

		for (int i = 0; i < *count; ++i) {
			*sum += arrayInt[i];
		}
	}

	__global__
	void fun0Gpu(int* count, int* arrayInt, int* arrayIntResult) {
		int val = 0;

		for (int i = 0; i < *count; ++i) {
			arrayIntResult[i] = arrayInt[i] * arrayInt[i];
		}
	}

	__global__
	void fun1Gpu(int* count, int* arrayInt, int* sum) {
		*sum = 0;

		for (int i = 0; i < *count; ++i) {
			*sum += arrayInt[i];
		}
	}
}

void CUDA_Test::Run() {
	printf("Test::Run begin.\n");
	
	int count = 1000;
	std::vector<int> ints;
	ints.reserve(count);

	for (int i = 0; i < count; ++i) {
		ints.emplace_back(i);
	}

	// CPU
	int sumCPU;
	{
		std::vector<int> resultInts;
		resultInts.resize(count, 0);

		fun0Cpu(&count, ints.data(), resultInts.data());
		fun1Cpu(&count, resultInts.data(), &sumCPU);
	}

	// GPU
	int sumGPU;
	{
		int* devCount;
		int* devSum;
		int* devArrayInt;
		int* devArrayIntResult;

		std::vector<int> resultInts;
		resultInts.resize(count, 0);

		cudaMalloc(&devCount,                  sizeof(int));
		cudaMalloc(&devSum,                    sizeof(int));
		cudaMalloc(&devArrayInt,       count * sizeof(int));
		cudaMalloc(&devArrayIntResult, count * sizeof(int));

		cudaMemcpy(devCount,         &count, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devSum,          &sumGPU, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devArrayInt, ints.data(), count * sizeof(int), cudaMemcpyHostToDevice);

		fun0Gpu <<<1, 1>>> (devCount, devArrayInt, devArrayIntResult);

		cudaMemcpy(&count, devCount, sizeof(int), cudaMemcpyDeviceToHost);

		fun1Gpu <<<1, 1>>> (devCount, devArrayIntResult, devSum);

		cudaMemcpy(&sumGPU,  devSum, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(resultInts.data(), devArrayIntResult, count * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(devCount);
		cudaFree(devSum);
		cudaFree(devArrayInt);
		cudaFree(devArrayIntResult);
	}

	//...
	if (sumCPU == sumGPU) {
		printf("Test::Run result OK [%i, %i].\n", sumCPU, sumGPU);
	}
	else {
		printf("Test::Run result FAIL [%i, %i].\n", sumCPU, sumGPU);
	}

	printf("Test::Run end.\n");
}

#else

	void CUDA_Test::Run() { }

#endif
