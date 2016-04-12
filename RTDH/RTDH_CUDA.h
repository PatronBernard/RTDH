#ifndef RTDH_CUDA_H
#define RTDH_CUDA_H

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "RTDH_helper_cuda.h"
#include "cuFFT_helper_functions.h"

typedef float2 Complex;

#define printCufftError() fprintf(stderr, "%s: line %d: %s \n", __FILE__, __LINE__, cufftStrError(result));

void findCUDAGLDevices();
#endif