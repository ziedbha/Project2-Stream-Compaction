#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Efficient {
		int* dev_bufData;

		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernEfficientScanUpsweep(int n, int level, int* g_data) {
			int index = threadIdx.x + blockIdx.x * blockDim.x;
		
			if (index >= n) {
				return;
			}

			// up-sweep
			int offset = 1 << (level + 1);
			int k = index * offset;
			if (k < n) {
				int halfOffset = 1 << level;
				g_data[k + offset - 1] += g_data[k + halfOffset - 1];
			}
		}

		__global__ void kernEfficientScanDownsweep(int n, int level, int* g_data) {
			int index = threadIdx.x + blockIdx.x * blockDim.x;

			if (index >= n) {
				return;
			}

			int offset = 1 << (level + 1);
			int k = index * offset;
			if (k < n) {
				int halfOffset = 1 << level;
				int temp = g_data[k + halfOffset - 1];
				g_data[k + halfOffset - 1] = g_data[k + offset - 1];
				g_data[k + offset - 1] += temp;
			}
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			int rounded = ilog2ceil(n);
			int nOriginal = n;
			n = 1 << rounded;

			// malloc device buffer
			cudaMalloc((void**)&dev_bufData, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");

			// copy input data into device buffer
			cudaMemcpy(dev_bufData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("CUDA Memcpy error!");

			timer().startGpuTimer();
			

			// upsweep
			for (int level = 0; level < ilog2(n); level++) {
				dim3 numberOfBlocks((n / (1 << (level + 1)) + BLOCK_SIZE - 1) / BLOCK_SIZE);
				kernEfficientScanUpsweep << <numberOfBlocks, BLOCK_SIZE >> > (n, level, dev_bufData);
				checkCUDAError("CUDA Upsweep error!");
				cudaDeviceSynchronize();
				checkCUDAError("CUDA Sync error!");
			}

			// downsweep
			int zero = 0;
			cudaMemcpy(dev_bufData + n - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
			for (int level = ilog2(n) - 1; level >= 0; level--) {
				dim3 numberOfBlocks((n / (1 << (level + 1)) + BLOCK_SIZE - 1) / BLOCK_SIZE);
				kernEfficientScanDownsweep << <numberOfBlocks, BLOCK_SIZE >> > (n, level, dev_bufData);
				checkCUDAError("CUDA Downsweep error!");
				cudaDeviceSynchronize();
				checkCUDAError("CUDA Sync error!");
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_bufData, nOriginal * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("CUDA Memcpy error!");

			cudaFree(dev_bufData);
		}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int *odata, const int *idata) {
			timer().resetGpuTimer();

			// Malloc on GPU: bools, input, and output
			int* dev_bools;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");

			int* dev_in;
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("CUDA Memcpy error!");

			int* dev_out;
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");

			dim3 numberOfBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

			// Bools: compute on GPU, copy to CPU to call scan() later
			timer().startGpuTimer();
			Common::kernMapToBoolean << <numberOfBlocks, BLOCK_SIZE >> > (n, dev_bools, dev_in);
			timer().endGpuTimer();
			int* bools = new int[n];
			cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("CUDA Memcpy error!");

			// Scan: alloc on CPU, call scan(), alloc on GPU
			int* scanned = new int[n];
			scan(n, scanned, bools);
			int size = bools[n - 1] == 0 ? scanned[n - 1] : scanned[n - 1] + 1; // number of true elements
			int* dev_scanned;
			cudaMalloc((void**)&dev_scanned, n * sizeof(int));
			checkCUDAError("CUDA Malloc error!");
			cudaMemcpy(dev_scanned, scanned, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("CUDA Memcpy error!");
			delete[] scanned;

			// Scatter
			timer().startGpuTimer();
			Common::kernScatter << <numberOfBlocks, BLOCK_SIZE >> > (n, dev_out, dev_in, dev_bools, dev_scanned);
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("CUDA Memcpy error!");

			delete[] bools;
			cudaFree(dev_bools);
			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(dev_scanned);
			return size;
		}
	}
}
