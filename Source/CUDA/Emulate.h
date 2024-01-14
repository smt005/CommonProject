#pragma once
#include <functional>

namespace CUDA_TEST {

// threadIdx.x
struct ThreadIdx {
	unsigned int x = 0;
	ThreadIdx(unsigned int _x) :x(_x) {}
};

// blockIdx.x
struct BlockIdx {
	unsigned int x = 0;
	BlockIdx(unsigned int _x) :x(_x) {}
};

// blockDim.x
struct BlockDim {
	unsigned int x = 0;
	BlockDim(unsigned int _x) :x(_x) {}
};

struct Index {
	BlockIdx blockIdx;
	ThreadIdx threadIdx;
	BlockDim blockDim;

	Index(unsigned int b, unsigned int t, unsigned int d)
		: blockIdx(b)
		, threadIdx(t)
		, blockDim(d)
	{}
};

class CUDA_Emulate final {
public:
	CUDA_Emulate(unsigned int countBlock, unsigned int countThread, std::function<void(Index)> fun);

public:
	template <typename T>
	static void cudaMalloc(T** ptrPtr, unsigned int size) {
		(*ptrPtr) = new T[size];
	}

	template <typename TypeCpy>
	static void cudaMemcpy(void* dst, void* src, size_t  size, TypeCpy type) {
		if (static_cast<unsigned int>(type) == 1) // cudaMemcpyHostToDevice
		{
			memcpy(dst, src, size);
		}
		else
			if (static_cast<unsigned int>(type) == 2) // cudaMemcpyDeviceToHost
			{
				memcpy(dst, src, size);
			}
	}

	template <typename T>
	static void cudaFree(T* ptrPtr) {
		delete[] ptrPtr;
		ptrPtr = nullptr;
	}

public:
	static unsigned int cudaMemcpyHostToDevice;
	static unsigned int cudaMemcpyDeviceToHost;
};

void Test();

}
