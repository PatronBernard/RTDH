#ifndef _KERNELS_H_
#define _KERNELS_H_
typedef float2 Complex; 

#include "device_launch_parameters.h"
#include "cuda.h"
#include <cuda_runtime_api.h>
#include "cufftXt.h"

//Calculates pointwise complex magnitude times a scalingfactor: scalingFactor*|z|
extern "C" void launch_cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z,
											   float scalingFactor, const int M, const int N);
//Calculates pointwise complex phase times a scalingfactor: scalingFactor*arg(z)
extern "C" void launch_cufftComplex2PhaseF(float* vbo_mapped_pointer, Complex *z,
											   float scalingFactor, const int M, const int N);
//Pointwise multiplication of the image by a checkerboard with alternating values -1, +1, -1, +1 ... 
extern "C" void launch_checkerBoard(Complex* A, int M, int N); 
//Pointwise multiplication of A and B and write the result to C
extern "C" void launch_matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N);
//Converts an unsigned char matrix to a strictly real (but Complex type) matrix normalized between 0 and 1
extern "C" void launch_unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N);
//Pointwise addition of two complex matrices
extern "C" void launch_addComplexPointWiseF(Complex *A, Complex *B, Complex *C, int M, int N);
//Pointwise addition of a constant, perhaps extend with scaling?
extern "C" void launch_addConstant(float* A, float c, int M, int N);
//Pointwise calculation of C=a*A+b*B+c
extern "C" void launch_linearCombination(Complex *A, Complex *B, Complex *C,float a, float b, float c, int M, int N);
//Pointwise calculation of a*(arg(A)-arg(B))+b
extern "C" void launch_phaseDifference(Complex *A, Complex *B, float *C, float a, float b, int M, int N);

extern "C" void launch_filterPhase(float *A, float *B, int windowsize, int M, int N);

extern "C" void launch_rescaleAndShiftF(float *A, float a, float b, int M, int N);
#endif