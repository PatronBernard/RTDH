//Fix some annoying warnings
#define _CRT_SECURE_NO_DEPRECATE

//GLEW
#define GLEW_STATIC
#include <GL\glew.h>

//GLFW
#include <GLFW\glfw3.h>

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

void mainLoop(GLFWwindow* window, reconParameters parameters, cudaGraphicsResource *cuda_vbo_resource);
__global__ void simple_vbo_kernel(float4 *pos, const int width, const int height);

//TODO: -fix everything, reorganize headers so it makes sense
//		-perhaps generate a test hologram at a smaller size?
int main(){

	//=========================INITIALIZATION==========================
	//Redirect stderror to log.txt.
	FILE* logfile = freopen("log.txt", "w", stderr);
	printTime(logfile);

	//Read the reconstruction parameters. 
	reconParameters parameters;
	read_parameters("parameters.txt", &parameters);

	//Initialize the GLFW window
	GLFWwindow *window = initGLFW(parameters.N, parameters.M); 

	//Set a few callbacks
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetKeyCallback(window, key_callback);

	//Search for CUDA devices and pick the best-suited one. 
	findCUDAGLDevices();

	//Read the recorded hologram from a file. This will be replaced by the CCD later on.
	Complex* h_recorded_hologram = (Complex*)malloc(sizeof(Complex)*parameters.N*parameters.M);
	if (h_recorded_hologram == NULL){ printError(); exit(EXIT_FAILURE); }
	float* h_recorded_hologram_real = read_data("recorded_hologram_scaled.bin");

	for (int i = 0; i < parameters.M*parameters.N; i++){
		h_recorded_hologram[i].x =  h_recorded_hologram_real[i];
		h_recorded_hologram[i].y = 0.0;
	}

	//We'll use a vertex array object with two VBO's. The first will house the vertex positions, the second will 
	//house their colours/complex value. We cannot put the positions and complex values in a single VBO because cuFFT requires
	//a float2. 
	
	GLuint vao;
	GLuint vbo[2];
	
	//Create the vertex array object and two vertex buffer object names.
	glGenVertexArrays(1, &vao);
	checkGLError(glGetError());
	
	glBindVertexArray(vao);
	checkGLError(glGetError());

	glGenBuffers(2, vbo);
	checkGLError(glGetError());

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	checkGLError(glGetError());

	
	//Calculate the position of each vertex (one for every pixel in the image). 
	float u, v, x, y;
	int k = 0;

	
	float *position = (float *) malloc(parameters.N*parameters.M * 2 * sizeof(float));
	for (int i = 0; i < parameters.N; i++){
		for (int j = 0; j < parameters.M; j++){
			u = (float)i - 0.5f*(float)parameters.N;
			v = (float)j - 0.5f*(float)parameters.M;
			x = (u) / (0.5f*(float)parameters.N);
			y = (v) / (0.5f*(float)parameters.M);

			position[k] = x;
			position[k + 1] = y;
			k += 2;
		}
	}
	
	//Load these vertex coordinates into the first vbo
	glBufferData(GL_ARRAY_BUFFER, parameters.N*parameters.M * 2 * sizeof(GLfloat), position, GL_DYNAMIC_DRAW);
	checkGLError(glGetError());

	glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,0);
	checkGLError(glGetError());

	glEnableVertexAttribArray(0);
	checkGLError(glGetError());

	//Bind the second VBO that will contain the complex data, register it as a CUDA resource so we can modify it.
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	checkGLError(glGetError());

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	checkGLError(glGetError());

	glEnableVertexAttribArray(0);
	checkGLError(glGetError());


	//This doesn't work, h_recorded_hologram is an array of structs with x- and y- fields, and glBufferData expects an array of the form 
	// x0 y0 | x1 y1 | ... | xn yn
	glBufferData(GL_ARRAY_BUFFER, parameters.N*parameters.M * 2 * sizeof(GLfloat), h_recorded_hologram, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkGLError(glGetError());

	cudaGraphicsResource *cuda_vbo_resource;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo[1], cudaGraphicsMapFlagsWriteDiscard));

	//Compile vertex and fragment shaders

	initShaders();
	checkGLError(glGetError());

	
	//=========================MAIN LOOP==========================

	mainLoop(window, parameters, cuda_vbo_resource);
	
	glfwTerminate();

	free(position);
	free(h_recorded_hologram);

	fprintf(stderr, "No errors (that I'm aware of)! \n");
	fclose(logfile);

	return 0;
};

void mainLoop(GLFWwindow* window,reconParameters parameters,cudaGraphicsResource *cuda_vbo_resource){
	// Measure frametime
	double frameTime = 0.0;
	int fps = 1;
	int fps_prev = 1;
	int framecounter = 1;
	std::string wtitle;

	while (!glfwWindowShouldClose(window))
	{
		glfwSetTime(0.0);
		float ratio = (float) parameters.N / (float) parameters.M;
		// handle events
		/*
		//Calculate position with CUDA
		// map OpenGL buffer object for writing from CUDA
		float4 *dptr; //This will become a float2 as to be compatible with cuFFT
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
			cuda_vbo_resource));
		//Run kernel, this will become a simple kernel and cufftExecC2C call
		//First thing to do: get an external stream to 
		dim3 block(8, 8, 1);
		dim3 grid((unsigned int) parameters.N / block.x, (unsigned int) parameters.N / block.y, 1);
		simple_vbo_kernel<<<grid,block>>>(dptr,parameters.N,parameters.M);
		//checkCudaErrors(cudaGetLastError());

		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
		*/

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

__global__ void simple_vbo_kernel(float4 *pos, const int width, const int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = (float)x / (float)width;
	float v = (float)y / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	float w = 0.5f*sqrt(pow(u, 2.0f) + pow(v, 2.0f));

	// write output vertex
	pos[y*width + x] = make_float4(u, v, u, v);
}
