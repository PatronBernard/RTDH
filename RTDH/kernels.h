#ifndef _KERNELS_H_
#define _KERNELS_H_
typedef float2 Complex; 

#include "device_launch_parameters.h"
#include "cuda.h"
#include <cuda_runtime_api.h>
#include "cufftXt.h"

extern "C" void launch_cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z, const int M, const int N);
extern "C" void launch_checkerBoard(Complex* A, int M, int N); 
extern "C" void launch_matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N);
extern "C" void launch_unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N);
extern "C" void launch_Modify(unsigned char *A, int M, int N);
#endif