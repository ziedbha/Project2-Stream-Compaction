#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
	namespace Thrust {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			thrust::host_vector<int> t_hostIn(idata, idata + n);
			thrust::host_vector<int> t_hostOut(odata, odata + n);

			thrust::device_vector<int> t_devIn(t_hostIn);
			thrust::device_vector<int> t_devOut(t_hostOut);

			timer().startGpuTimer();
			thrust::exclusive_scan(t_devIn.begin(), t_devIn.end(), t_devOut.begin());
			timer().endGpuTimer();

			t_hostOut = t_devOut;

			// copy memory from host vector --> odata
			memcpy(odata, &t_hostOut[0], n * sizeof(int));
		}
	}
}
