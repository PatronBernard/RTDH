#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufftXt.h"


typedef float2 Complex; 
/*
__global__ void cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z, const int M, const int N){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
	float magnitude = sqrt(pow(z[i*N + j].x, (float)2) + pow(z[i*N + j].y, (float)2));
	vbo_mapped_pointer[i*N + j] = magnitude;// log(1.0 + magnitude);// / sqrt((float)M*(float)N)) / 75.0; //This is a constant so we might want to calculate this beforehand. 
	}
};

__global__ void matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		C[i*N + j].x = A[i*N + j].x*B[i*N + j].x;
		C[i*N + j].y = A[i*N + j].y*B[i*N + j].y;		
	}
}
*/
__global__ void checkerBoard(Complex* A, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		A[i*N + j].x = A[i*N + j].x*(float)((i + j) % 2) -A[i*N + j].x*(float)(1 - ((i + j) % 2));
		A[i*N + j].y = A[i*N + j].y*(float)((i + j) % 2) -A[i*N + j].y*(float)(1 - ((i + j) % 2));
	}
}

extern "C"
void launch_checkerBoard(Complex* A, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	checkerBoard<<<grid, block>>>(A,M,N);
}

/*
__global__ void unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		z[i*N+j].x=(float) A[i*N+j]/255.0;
		z[i*N+j].y=0.0;
	}
}

extern "C"
void launch_cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z, const int M, const int N){
	//Set up the grid
	dim3 block(16, 16, 1);
	//I added the +1 because it might round down which can mean that not all pixels are processed in each kernel. 
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	cufftComplex2MagnitudeF<<<grid, block>>>(vbo_mapped_pointer, z, M, N);
}  

extern "C"
void launch_matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	matrixMulComplexPointw<<<grid, block>>>(A, B, C, M, N);
}


extern "C"
void launch_unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	unsignedChar2cufftComplex<<<grid, block>>>(z, A, M, N);
}
*/