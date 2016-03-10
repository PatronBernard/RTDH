#ifndef RTDH_CUDA_H
#define RTDH_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include "helper_cuda.h"
#include "cuFFT_helper_functions.h"

typedef float2 Complex;

#define printCufftError()	fprintf(stderr, "%s: line %d: %s \n", __FILE__, __LINE__, cufftStrError(result));

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