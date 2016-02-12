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

//TODO: fix everything, reorganize headers so it makes sense
int main(){
	//Redirect stderror to log.txt
	FILE* logfile = freopen("log.txt", "w", stderr);
	//Redirect stderror to log.txt
	logfile = freopen("log.txt", "w", stderr);
	printTime(logfile);

	
	reconParameters parameters;
	read_parameters("parameters.txt", &parameters),

	initGLFW(parameters.N, parameters.M); 

	findCUDAGLDevices();


	//MAIN LOOP
	/*
	// Measure frametime by calculating the time elapsed since the last frame
	double frameTime = 0.0;
	int fps = 1;
	int fps_prev = 1;
	int framecounter = 1;
	std::string wtitle; //The FPS will be displayed in the window title.

	while (!glfwWindowShouldClose(window))
	{
		glfwSetTime(0.0);
		float ratio = width / (float)height;
		// handle events

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
		dim3 grid((unsigned int)width / block.x, (unsigned int)height / block.y, 1);
		//simple_vbo_kernel<<<grid,block>>>(dptr,width,height);
		//checkCudaErrors(cudaGetLastError());

		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));


		glDrawArrays(GL_POINTS, 0, width*height);

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
	*/
	glfwTerminate();
	return 0;
};