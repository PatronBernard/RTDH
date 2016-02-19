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

void createVBOCUDA(GLuint *vbo, int element_components_no, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, int width, int height){
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = width * height * element_components_no * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW); //Later on use glBufferSubData!

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
	checkGLError(glGetError());
}
#endif