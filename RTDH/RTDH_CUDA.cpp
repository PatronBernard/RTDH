#include "RTDH_CUDA.h"

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
};

