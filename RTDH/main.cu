//Fix some annoying warnings
#define _CRT_SECURE_NO_DEPRECATE

//GLEW
#define GLEW_STATIC
#include <GL\glew.h>

//GLM
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Project specific includes
#include <cuda_gl_interop.h>//Visualization
#include "cufftXt.h"		//CUDA FFT
#include "helper_cuda.h"	//heckCudaErrors

#include "RTDH_utility.h"	
#include "RTDH_GLFW.h"
#include "RTDH_CUDA.h"

//Vimba stuff
#include "ApiController.h"

#include "LoadSaveSettings.h"

//Other
#include <iostream>

//GLFW
#include <GLFW\glfw3.h>

//Pointwise multiplication of two Complex arrays
__global__ void matrixMulComplexPointw(Complex* A, Complex* B, Complex* C, int M, int N);

//Transforms cufftComplex into a float by calculating the magnitude
__global__ void cufftComplex2MagnitudeF(float *vbo_dptr, Complex *z, const int M, const int N, Complex* d_isNan);

//Checkerboard function that ensures the quadrants are in the right place after fourier-transforming.
__global__ void checkerBoard(Complex* A,const int M,const int N);

//Transforms raw camera input into a normalized float. This will change later on
__global__ void unsignedChar2cufftComplex(unsigned char *A,int M, int N);


//This getting out of hand, just add its contents to main() ffs !
void mainLoop(	GLFWwindow* window, 
				GLuint shaderprogram, 
				GLuint projection_Handle, 
				reconParameters parameters, 
				cudaGraphicsResource *cuda_vbo_resource, 
				Complex* d_recorded_hologram, 
				Complex* d_chirp,
				Complex* d_propagated,
				cufftHandle plan,
				cufftResult result,
				std::string strCameraID,
				AVT::VmbAPI::Examples::ApiController apiController,
				AVT::VmbAPI::FramePtr pFrame,
				unsigned char* d_recorded_hologram_uchar);

#define PI	3.1415926535897932384626433832795028841971693993751058209749
#define PI2 1.570796326794896619231321691639751442098584699687552910487


int main(){
	//Redirect stderror to log.txt.
	FILE* logfile = freopen("log.txt", "w", stderr);
	printTime(logfile);

	//Initialize the Vimba API and print some info.
	AVT::VmbAPI::Examples::ApiController apiController;

	std::cout << "Vimba Version V " << apiController.GetVersion() << "\n";

	//Start the API
	VmbErrorType vmb_err = VmbErrorSuccess;
	vmb_err = apiController.StartUp();
	if(vmb_err != VmbErrorSuccess){
		fprintf(stderr,"%s: line %d: Vimba API Error: apiController.Startup() failed. \n",__FILE__,__LINE__);
		exit(EXIT_FAILURE); 
	}
	
	//Look for cameras
	std::string strCameraID;
	AVT::VmbAPI::CameraPtr pCamera;
	AVT::VmbAPI::CameraPtrVector cameraList = apiController.GetCameraList();
	if(cameraList.size() == 0){
		fprintf(stderr,"Error: couldn't find a camera. Shutting down... \n");
		apiController.ShutDown();
		exit(EXIT_FAILURE);
	}
	else{
		//If a camera is found, store its pointer.
		pCamera=cameraList[0];
		vmb_err = pCamera->GetID(strCameraID);
		if(vmb_err != VmbErrorSuccess){
			printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);}

		//Open the camera and load its settings.
		vmb_err = pCamera->Open(VmbAccessModeFull);
		AVT::VmbAPI::StringVector loadedFeatures;
        AVT::VmbAPI::StringVector missingFeatures;
        vmb_err = AVT::VmbAPI::Examples::LoadSaveSettings::LoadFromFile(pCamera, "CameraSettings.xml", loadedFeatures, missingFeatures, false);
		if(vmb_err != VmbErrorSuccess){
				printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);}
		vmb_err = pCamera->Close();
		if(vmb_err != VmbErrorSuccess){
				printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);}
	}
	
	//This will be removed later on.
	AVT::VmbAPI::FramePtr pFrame;
	vmb_err = apiController.AcquireSingleImage(strCameraID, pFrame);
	if(vmb_err != VmbErrorSuccess){
		printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);}

	
	//=========================INITIALIZATION==========================
	
	//Read the reconstruction parameters. 
	reconParameters parameters;
	read_parameters("parameters.txt", &parameters);

	VmbUint32_t frameWidth = 0;
	VmbUint32_t frameHeight = 0;
	pFrame->GetWidth(frameWidth);
	pFrame->GetHeight(frameHeight);

	//Override the parameters supplied in the file (which will be obsolete anyway); 
	parameters.M=frameHeight;
	parameters.N=frameWidth;

	//Initialize the GLFW window
	//GLFWwindow *window = initGLFW(parameters.N, parameters.M); 
	GLFWwindow *window = initGLFW((int)parameters.N/4, (int) parameters.M/4); 

	//Set a few callbacks
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetKeyCallback(window, key_callback);

	//Search for CUDA devices and pick the best-suited one. 
	findCUDAGLDevices();

	//Allocate and set up the chirp-function, copy it to the GPU memory.
	Complex* h_chirp = (Complex*)malloc(sizeof(Complex)*parameters.N*parameters.M);
	if (h_chirp == NULL){ printError(); exit(EXIT_FAILURE); }

	construct_chirp(h_chirp, parameters.M, parameters.N, parameters.lambda, parameters.rec_dist, parameters.pixel_y, parameters.pixel_x);

	Complex* d_chirp;
	checkCudaErrors(cudaMalloc((void**)&d_chirp, sizeof(Complex)*parameters.M*parameters.N));

	checkCudaErrors(cudaMemcpy(d_chirp, h_chirp, sizeof(Complex)*parameters.M*parameters.N, cudaMemcpyHostToDevice));
	
	//Set up the grid
	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)parameters.M / block.x+1, (unsigned int)parameters.N / block.y+1, 1);

	checkerBoard << <grid, block >> >(d_chirp, parameters.M, parameters.N);
	checkCudaErrors(cudaGetLastError());

	//Read the recorded hologram from a file. This will be replaced by the CCD later on.
	Complex* h_recorded_hologram = (Complex*)malloc(sizeof(Complex)*parameters.M*parameters.N);
	if (h_recorded_hologram == NULL){ printError(); exit(EXIT_FAILURE); }



	//The binary must be in single-precision row-major order !!!
	float* h_recorded_hologram_real = read_data("recorded_hologram.bin");

	for (int i = 0; i < parameters.M*parameters.N; i++){
		h_recorded_hologram[i].x =  h_recorded_hologram_real[i];
		h_recorded_hologram[i].y = 0.0;
	}
	
	//Copy the hologram to the GPU
	Complex* d_recorded_hologram;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram, sizeof(Complex)*parameters.M*parameters.N));

	checkCudaErrors(cudaMemcpy(d_recorded_hologram,h_recorded_hologram,sizeof(Complex)*parameters.M*parameters.N,cudaMemcpyHostToDevice));

	unsigned char* d_recorded_hologram_uchar;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram_uchar,sizeof(unsigned char)*parameters.M*parameters.N));

	//Copy the recorded image to the device
	VmbUchar_t *pImage;
	vmb_err = pFrame->GetImage(pImage);
	checkCudaErrors(cudaMemcpy(	d_recorded_hologram_uchar,pImage,
								sizeof(unsigned char)*parameters.M*parameters.N,
								cudaMemcpyHostToDevice));

	Complex* d_propagated;
	checkCudaErrors(cudaMalloc((void**)&d_propagated, sizeof(Complex)*parameters.M*parameters.N));


	//We'll use a vertex array object with two VBO's. The first will house the vertex positions, the second will 
	//house the magnitude that will be calculated with a kernel. 
	
	GLuint vao;
	GLuint vbo[2];
	
	//Create the vertex array object and two vertex buffer object names.
	glGenVertexArrays(1, &vao);
	checkGLError(glGetError());
	
	glBindVertexArray(vao);
	checkGLError(glGetError());

	glGenBuffers(2, vbo);
	checkGLError(glGetError());

	//First let's set up all vertices in the first vbo. 
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	checkGLError(glGetError());

	
	//Calculate the position of each vertex (one for every pixel in the image). 
	float u, v, x, y;
	int k = 0;

	
	float *vertices = (float *) malloc(parameters.M*parameters.N * 2 * sizeof(float));
	
	for (int i = 0; i < parameters.M; i++){
		for (int j = 0; j < parameters.N; j++){
			u = (float)j - 0.5f*(float)parameters.N;
			v = (float)i - 0.5f*(float)parameters.M;
			x = (u) / (0.5f*(float)parameters.N);
			y = -(v) / (0.5f*(float)parameters.M);

			vertices[k] = x;
			vertices[k + 1] = y;
			k += 2;
		}
	}
	
	//Load these vertex coordinates into the first vbo
	glBufferData(GL_ARRAY_BUFFER, parameters.N*parameters.M * 2 * sizeof(GLfloat), vertices, GL_DYNAMIC_DRAW);
	checkGLError(glGetError());

	glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,0);
	checkGLError(glGetError());

	glEnableVertexAttribArray(0);
	checkGLError(glGetError());

	//Bind the second VBO that will contain the magnitude of each complex number. 
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	checkGLError(glGetError());

	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	checkGLError(glGetError());

	glEnableVertexAttribArray(1);
	checkGLError(glGetError());

	//IDEE: schrijf een kernel van cufftcomplex - > VBO
	//This doesn't work, h_recorded_hologram is an array of structs with x- and y- fields, and glBufferData expects an array of the form 
	// x0 y0 | x1 y1 | ... | xn yn

	//This is the VBO that the complex magnitudes will be written to for visualization.
	glBufferData(GL_ARRAY_BUFFER, parameters.M*parameters.N * 1 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkGLError(glGetError());

	cudaGraphicsResource *cuda_vbo_resource;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo[1], cudaGraphicsMapFlagsWriteDiscard));

	//Compile vertex and fragment shaders

	GLuint shaderprogram = initShaders();
	checkGLError(glGetError());

	// Set up cuFFT stuff
	cufftComplex* d_reconstructed;
	cudaMalloc((void**)&d_reconstructed, sizeof(cufftComplex)*parameters.M*parameters.N);

	//Set up plan
	cufftResult result = CUFFT_SUCCESS;
	cufftHandle plan;
	result = cufftPlan2d(&plan, parameters.M, parameters.N, CUFFT_C2C);
	if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
	
	//=========================MAIN LOOP==========================

	GLuint projection_Handle= glGetUniformLocation(shaderprogram, "Projection");

	mainLoop(	window, 
				shaderprogram, 
				projection_Handle, 
				parameters, 
				cuda_vbo_resource, 
				d_recorded_hologram, 
				d_chirp,
				d_propagated, 
				plan,
				result,
				strCameraID,
				apiController,
				pFrame,
				d_recorded_hologram_uchar);

	//Export the last reconstructed frame. 
	Complex* h_reconstructed=(Complex*) malloc(sizeof(Complex)*parameters.M*parameters.N);
	checkCudaErrors(cudaMemcpy(h_reconstructed, d_propagated, sizeof(Complex)*parameters.M*parameters.N, cudaMemcpyDeviceToHost));

	export_complex_data("reconstructed_hologram.bin", h_reconstructed, parameters.M*parameters.N);
	

	//Cleanup

	
	checkCudaErrors(cudaFree(d_recorded_hologram));
	checkCudaErrors(cudaFree(d_recorded_hologram_uchar));
	checkCudaErrors(cudaFree(d_chirp));
	checkCudaErrors(cudaFree(d_propagated));

	

	free(vertices);
	free(h_recorded_hologram);
	free(h_chirp);
	free(h_reconstructed);
	
	glfwTerminate();

	//Gracefully (lol, as if) end Vimba stuff.
	if(pCamera != NULL){pCamera->Close();}
	apiController.ShutDown();

	
	fprintf(stderr, "No errors (that I'm aware of)! \n");
	fclose(logfile);

	return 0;
};

//Not needed 
__global__ void restrictToRange(float* vbo_magnitude, const int M, const int N){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		float min_Val = 5.7*7000.0;
		float max_Val = 1.11*70000.0;
		vbo_magnitude[i*N + j] = (vbo_magnitude[i*N + j] - min_Val) / (max_Val - min_Val); //This is a constant so we might want to calculate this beforehand. 
	}
};

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

__global__ void checkerBoard(Complex* A, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		A[i*N + j].x = A[i*N + j].x*(float)((i + j) % 2) -A[i*N + j].x*(float)(1 - ((i + j) % 2));
		A[i*N + j].y = A[i*N + j].y*(float)((i + j) % 2) -A[i*N + j].y*(float)(1 - ((i + j) % 2));
	}
}

__global__ void unsignedChar2cufftComplex(Complex* z, unsigned char *A, int M, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < M && j < N){
		z[i*N+j].x=(float) A[i*N+j]/255.0;
		z[i*N+j].y=0.0;
	}
};



void mainLoop(	GLFWwindow* window, 
				GLuint shaderprogram, 
				GLuint projection_Handle, 
				reconParameters parameters, 
				cudaGraphicsResource *cuda_vbo_resource, 
				Complex* d_recorded_hologram, 
				Complex* d_chirp,
				Complex* d_propagated,
				cufftHandle plan,
				cufftResult result,
				std::string strCameraID,
				AVT::VmbAPI::Examples::ApiController apiController,
				AVT::VmbAPI::FramePtr pFrame,
				unsigned char* d_recorded_hologram_uchar){

	// Measure frametime
	double frameTime = 0.0;
	int fps = 1;
	int fps_prev = 1;
	int framecounter = 1;
	std::string wtitle;
	VmbErrorType vmb_err;
	VmbUchar_t *pImage;

	while (!glfwWindowShouldClose(window))
	{
		//Start measuring frame time
		glfwSetTime(0.0);
		float ratio = (float)parameters.N / (float)parameters.M;
		
		//Set up the grid
		dim3 block(16, 16, 1);
		//I added the +1 because it might round down which can mean that not all pixels are processed in each kernel. 
		dim3 grid((unsigned int)parameters.M / block.x+1, (unsigned int)parameters.N / block.y+1, 1);
		
		//Fetch an image, copy it to the device and convert it
 
		vmb_err = apiController.AcquireSingleImage(strCameraID, pFrame);
		if(vmb_err != VmbErrorSuccess){
		printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);}

		
		vmb_err = pFrame->GetImage(pImage);
		checkCudaErrors(cudaMemcpy(	d_recorded_hologram_uchar,pImage,
								sizeof(unsigned char)*parameters.M*parameters.N,
								cudaMemcpyHostToDevice));

		unsignedChar2cufftComplex<< <grid, block >> >(d_recorded_hologram,d_recorded_hologram_uchar,parameters.M,parameters.N);

		matrixMulComplexPointw << <grid, block >> >(d_chirp, d_recorded_hologram, d_propagated, parameters.M, parameters.N);
		checkCudaErrors(cudaGetLastError());

		result = cufftExecC2C(plan,d_propagated, d_propagated, CUFFT_FORWARD);
		if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
		
		float *vbo_mapped_pointer; //This is the pointer that we'll write the result to for display in OpenGL.
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));

		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vbo_mapped_pointer, &num_bytes, cuda_vbo_resource));


		cufftComplex2MagnitudeF << <grid, block >> >(vbo_mapped_pointer, d_recorded_hologram, parameters.M, parameters.N);

		//restrictToRange << <grid, block >> >(dptr, parameters.M, parameters.N);

		checkCudaErrors(cudaGetLastError());
		
		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
		
		int w_width, w_height;
		glfwGetWindowSize(window, &w_width, &w_height);
		
		//glm::mat4 Projection = glm::ortho(-(float)parameters.N / (float)w_width, (float)parameters.N / (float)w_width, -(float)parameters.M / (float)w_height, (float)parameters.M / (float)w_height);
		glm::mat4 Projection = glm::ortho(-1.0,1.0,-1.0,1.0);

		glUniformMatrix4fv(projection_Handle, 1, GL_FALSE, &Projection[0][0]);


		glDrawArrays(GL_POINTS, 0, parameters.N*parameters.M);

		glfwSwapBuffers(window);
		glfwPollEvents();

		//Display fps in cmd, not sure if averaging is that useful
		frameTime = glfwGetTime();
		fps_prev = fps;
		fps = (int)(0.5*(1. / frameTime + (float)fps_prev));

		//Update FPS every 15 frames
		framecounter += 1;
		if (framecounter == 15){
			framecounter = 1;
			fprintf(stdout, "\r Frames Per Second: %i             ", fps);
			wtitle = std::to_string(fps);
			glfwSetWindowTitle(window, wtitle.c_str());
		}
	}
};