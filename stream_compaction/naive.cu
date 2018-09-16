#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
		int* dev_bufIn;
		int* dev_bufOut;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
		__global__ void kernNaiveScan(int n, int level, int* g_odata, int* g_idata) {
			int index = threadIdx.x + blockIdx.x * blockDim.x;
			if (index >= n) {
				return;
			}
			int offset = (int)powf(2.0f, (float)(level - 1));
			if (index >= offset) {
				g_odata[index] = g_idata[index - offset] + g_idata[index];
			}
			else {
				g_odata[index] = g_idata[index];
			}
		}

		__global__ void kernInclusiveToExclusive(int n, int* g_odata, int* g_idata) {
			int index = threadIdx.x + blockIdx.x * blockDim.x;
			if (index >= n) {
				return;
			}

			if (index > 0) {
				g_odata[index] = g_idata[index - 1];
			}
			else {
				g_odata[0] = 0;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// malloc device buffers
			cudaMalloc((void**)&dev_bufIn, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");
			cudaMalloc((void**)&dev_bufOut, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");

			// copy input data into device
			cudaMemcpy(dev_bufIn, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("CUDA Memcpy error!");

			timer().startGpuTimer();
			int blockSize = 1024;
			dim3 numberOfBlocks((n + blockSize - 1) / blockSize);

			// placeholder pointers to do dual buffering
			int* input = dev_bufIn;
			int* output = dev_bufOut;
			for (int i = 1; i <= ilog2ceil(n); i++) {
				kernNaiveScan << <numberOfBlocks, blockSize >> >(n, i, output, input);
				
				// swap buffers
				int* temp = output;
				output = input;
				input = temp;

				cudaDeviceSynchronize();
			}
			timer().endGpuTimer();

			// the above scan is inclusive --> run the conversion kernel
			kernInclusiveToExclusive << <numberOfBlocks, blockSize >> >(n, output, input);
			if (output == dev_bufOut) {
				cudaMemcpy(odata, dev_bufOut, n * sizeof(int), cudaMemcpyDeviceToHost);
			}
			else {
				cudaMemcpy(odata, dev_bufIn, n * sizeof(int), cudaMemcpyDeviceToHost);
			}

			// free malloc'd device memory
			cudaFree(dev_bufIn);
			cudaFree(dev_bufOut);
		}
    }
}
