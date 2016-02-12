#ifndef RTDH_CUDA_H
#define RTDH_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include "helper_cuda.h"


void findCUDAGLDevices(){
	//Look for a CUDA device
	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	if (deviceCount > 0){
		checkCudaErrors(cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()));
	}
	else{
		fprintf(stderr, "Failed to find a CUDA device. Exiting... \n");
		exit(EXIT_FAILURE);
	}
}
#endif