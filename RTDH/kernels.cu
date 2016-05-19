#ifndef _KERNELS_CU_
#define _KERNELS_CU_
typedef float2 Complex; 

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cufftXt.h"
#include <math.h>
#include "globals.h"

__global__ void cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z, float scalingFactor, const int M, const int N){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
	float magnitude = sqrt(pow(z[i*N + j].x, (float)2) + pow(z[i*N + j].y, (float)2));
	vbo_mapped_pointer[i*N + j] = magnitude*scalingFactor; 
	}
};

__global__ void cufftComplex2PhaseF(float* vbo_mapped_pointer, Complex *z, float scalingFactor, const int M, const int N){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
	float phase = atan2(z[i*N + j].y,z[i*N + j].x);
	vbo_mapped_pointer[i*N + j] = phase*scalingFactor; 
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

__global__ void checkerBoard(Complex* A, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		A[i*N + j].x = A[i*N + j].x*(float)((i + j) % 2) -A[i*N + j].x*(float)(1 - ((i + j) % 2));
		A[i*N + j].y = A[i*N + j].y*(float)((i + j) % 2) -A[i*N + j].y*(float)(1 - ((i + j) % 2));
	}
}

__global__ void unsignedChar2cufftComplex(Complex* z, unsigned char*A, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		z[i*N+j].x=(float) A[i*N+j]/255.0;
		z[i*N+j].y=0.0;
	}
};

__global__ void addComplexPointWiseF(Complex *A, Complex *B, Complex *C,int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		C[i*N+j].x=A[i*N+j].x+B[i*N+j].x;
		C[i*N+j].y=A[i*N+j].y+B[i*N+j].y;
	}
}

__global__ void addConstant(float *A, float c, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		A[i*N+j]+=c;
	}
}

__global__ void linearCombination(Complex *A, Complex *B, Complex *C,float a, float b, float c, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		C[i*N+j].x=a*A[i*N+j].x+b*B[i*N+j].x+c;
		C[i*N+j].y=a*A[i*N+j].y+b*B[i*N+j].y+c;
	}
}

__global__ void phaseDifference(Complex *A, Complex *B, float *C, float a, float b, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		float phaseA = atan2f(A[i*N + j].y,A[i*N + j].x);
		float phaseB = atan2f(B[i*N + j].y,B[i*N + j].x);
		C[i*N + j]=a*fmodf(phaseA-phaseB,2*PI)+b;
	}
}

__global__ void rescaleAndShiftF(float* A, float a, float b, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		A[i*N + j] = a*A[i*N + j] + b;
	}
}

__global__ void sin(float* A, float* B, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		B[i*N + j] = sin(A[i*N + j]);
		
	}
}

__global__ void cos(float* A, float* B, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		B[i*N + j] = cos(A[i*N + j]);

	}
}

__global__ void filterPhase(float *Asin, float *Acos, float *B, int windowsize, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	//Relative index
	int u_lower = (int)floorf(-0.5*windowsize);
	int u_upper = u_lower + windowsize;
	int v_lower = (int)floorf(-0.5*windowsize);
	int v_upper = v_lower + windowsize;

	if (i < (M + u_lower) && i > u_upper && j < (N + v_lower) && j > v_upper){
		float avg_s = 0.0;
		float avg_c = 0.0;
		for (int u = i + u_lower; u <= i + u_upper; u++){
			for (int v = j + v_lower; v <= j + v_upper; v++){
				avg_s += Asin[u*N + v];
				avg_c += Acos[u*N + v];
			}
		}
		B[i*N + j] = atan2f(avg_s / ((float)windowsize*windowsize), avg_c / ((float)windowsize*windowsize));
	}
}

__global__ void constructChirp(Complex* A, 
								float rec_dist, 
								float lambda, 
								float pixel_x, float pixel_y, 
								int M, int N){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i < M && j < N){
		float a = PI / (rec_dist*lambda);
		float ch_exp, x, y;
		x = (float)j - (float)(N / 2.0);
		y = (float)i - (float)(M / 2.0);
		ch_exp = (float)pow(x*pixel_x, 2) + (float)pow(y*pixel_y, 2);
		A[i*N + j].x = cos(a*ch_exp);
		A[i*N + j].y = sin(a*ch_exp);
	}
}

extern "C"
void launch_cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z, float scalingFactor, const int M, const int N){
	//Set up the grid
	dim3 block(16, 16, 1);
	//I added the +1 because it might round down which can mean that not all pixels are processed in each kernel. 
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	cufftComplex2MagnitudeF<<<grid, block>>>(vbo_mapped_pointer, z, scalingFactor, M, N);
}  

extern "C"
void launch_cufftComplex2PhaseF(float* vbo_mapped_pointer, Complex *z,float scalingFactor, const int M, const int N){
	//Set up the grid
	dim3 block(16, 16, 1);
	//I added the +1 because it might round down which can mean that not all pixels are processed in each kernel. 
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	cufftComplex2PhaseF<<<grid, block>>>(vbo_mapped_pointer, z, scalingFactor, M, N);
}  

extern "C"
void launch_matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	matrixMulComplexPointw<<<grid, block>>>(A, B, C, M, N);
}

extern "C"
void launch_checkerBoard(Complex* A, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	checkerBoard<<<grid, block>>>(A,M,N);
}

extern "C"
void launch_unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	unsignedChar2cufftComplex<<<grid, block>>>(z, A, M, N);
}


extern "C" 
void launch_addComplexPointWiseF(Complex *A, Complex *B, Complex *C, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	addComplexPointWiseF<<<grid, block>>>(A, B, C, M, N);
}

extern "C" 
void launch_addConstant(float* A, float c, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	addConstant<<<grid, block>>>(A, c, M, N);
}

extern "C"
void launch_linearCombination(Complex *A, Complex *B, Complex *C,float a, float b, float c, int M, int N)
{
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	linearCombination<<<grid,block>>>(A, B, C, a, b, c, M, N);
}

extern "C"
void launch_phaseDifference(Complex *A, Complex *B, float *C, float a, float b, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	phaseDifference<<<grid,block>>>(A, B, C, a, b, M, N);
};
/*
extern "C"
void launch_filterPhase(float *A, float *B, int windowsize, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	filterPhase <<<grid, block >>>(A, B, windowsize, M, N);
};
*/
extern "C"
void launch_rescaleAndShiftF(float *A, float a, float b, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	rescaleAndShiftF<<<grid, block >>>(A, a, b, M, N);
};

extern "C"
void launch_sin(float *A, float *B, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	sin <<<grid, block >>>(A, B, M, N);
};

extern "C"
void launch_cos(float *A, float *B, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	cos<<<grid, block >>>(A, B, M, N);
};

extern "C"
void launch_filterPhase(float *Asin, float *Acos, float *B, int windowsize, int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	filterPhase <<<grid, block >>>(Asin, Acos, B, windowsize, M, N);
};

extern "C"
void launch_constructChirp( Complex* A,
							float rec_dist,
							float lambda,
							float pixel_x, float pixel_y,
							int M, int N){
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	constructChirp <<<grid, block >>>(A, rec_dist, lambda, pixel_x, pixel_y, M, N);
}
#endif