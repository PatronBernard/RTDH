typedef float2 Complex; 

__global__ void cufftComplex2MagnitudeF(float* vbo_mapped_pointer, Complex *z, const int M, const int N){;

__global__ void matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N);

__global__ void checkerBoard(Complex* A, int M, int N);

__global__ void unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N);

/*
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
*/