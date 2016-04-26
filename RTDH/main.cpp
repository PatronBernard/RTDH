//Fix some annoying warnings
#define _CRT_SECURE_NO_DEPRECATE

//GLEW
#define GLEW_STATIC
#include <GL\glew.h>

//GLM
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

//CUDA
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "kernels.h"
#include <cuda_gl_interop.h>//Visualization
#include <cufftXt.h>		//CUDA FFT
#include <cuda_runtime.h>

#include "RTDH_helper_cuda.h"	//heckCudaErrors
#include "RTDH_utility.h"	
#include "RTDH_GLFW.h"
#include "RTDH_CUDA.h"

//#include "ListFeatures\Source\ListFeatures.h"

//Other
#include <iostream>
#include "globals.h"

//GLFW
#include <GLFW\glfw3.h>

//Vimba stuff
#include "ApiController.h"
#include "LoadSaveSettings.h"

//imGUI Stuff
#include <imgui.h>
#include "imgui_impl_glfw.h"

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error %d: %s\n", error, description);
}

int main()
{

	//Redirect stderror to log.txt.
	FILE* logfile = freopen("log.txt", "w", stderr);
	printTime(logfile);

	//Initialize the Vimba API and print some info.
	AVT::VmbAPI::Examples::ApiController apiController;
	std::cout << "Vimba Version V " << apiController.GetVersion() << "\n";
	printConsoleInfo();
	//Start the API
	VmbErrorType vmb_err = VmbErrorSuccess;
	vmb_err = apiController.StartUp();

	if (vmb_err != VmbErrorSuccess){
		fprintf(stderr, "%s: line %d: Vimba API Error: apiController.Startup() failed. \n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Look for cameras
	std::string strCameraID;
	AVT::VmbAPI::CameraPtr pCamera;
	AVT::VmbAPI::CameraPtrVector cameraList = apiController.GetCameraList();
	if (cameraList.size() == 0){
		fprintf(stderr, "Error: couldn't find a camera. Shutting down... \n");
		apiController.ShutDown();
		exit(EXIT_FAILURE);
	}
	else{
		//If a camera is found, store its pointer.
		pCamera = cameraList[0];
		vmb_err = pCamera->GetID(strCameraID);
		if (vmb_err != VmbErrorSuccess){
			printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);
		}

		//Open the camera and load its settings.

		vmb_err = pCamera->Open(VmbAccessModeFull);
		AVT::VmbAPI::StringVector loadedFeatures;
		AVT::VmbAPI::StringVector missingFeatures;
		vmb_err = AVT::VmbAPI::Examples::LoadSaveSettings::LoadFromFile(pCamera, "CameraSettings.xml", loadedFeatures, missingFeatures, false);
		if (vmb_err != VmbErrorSuccess){
			printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);
		}
		vmb_err = pCamera->Close();
		if (vmb_err != VmbErrorSuccess){
			printVimbaError(vmb_err); apiController.ShutDown(); exit(EXIT_FAILURE);
		}

	}


	//Fetch the dimensions of the image.
	pCamera->Open(VmbAccessModeFull);

	AVT::VmbAPI::FeaturePtr feature_width;
	pCamera->GetFeatureByName("Width", feature_width);

	VmbInt64_t width;
	feature_width->GetValue(width);

	AVT::VmbAPI::FeaturePtr feature_height;
	pCamera->GetFeatureByName("Height", feature_height);

	VmbInt64_t height;
	feature_height->GetValue(height);
	pCamera->Close();

	int M = (int)height;
	int N = (int)width;

	//=========================INITIALIZATION==========================

	//Read the reconstruction parameters. 
	reconParameters parameters;
	read_parameters("parameters.txt", &parameters);

	//Initialize the GLFW window
	GLFWwindow *window = initGLFW((int)N / 2, (int)M / 2);
	glfwMakeContextCurrent(window);
	//glViewport(0, 0, 512, 512);


	//Set a few callbacks
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetKeyCallback(window, key_callback);

	//Search for CUDA devices and pick the best-suited one. 
	findCUDAGLDevices();

	// Setup ImGui binding
	ImGui_ImplGlfw_Init(window, true);

	// Allocate and set up the chirp - function, copy it to the GPU memory.Also checkerboard it
		// so we don't have to do that in the main loop.
		Complex* h_chirp = (Complex*)malloc(sizeof(Complex)*N*M);
	if (h_chirp == NULL){ printError(); exit(EXIT_FAILURE); }

	construct_chirp(h_chirp, M, N, parameters.lambda, parameters.rec_dist, parameters.pixel_y, parameters.pixel_x);

	Complex* d_chirp;
	checkCudaErrors(cudaMalloc((void**)&d_chirp, sizeof(Complex)*M*N));

	checkCudaErrors(cudaMemcpy(d_chirp, h_chirp, sizeof(Complex)*M*N, cudaMemcpyHostToDevice));


	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x + 1, (unsigned int)N / block.y + 1, 1);
	launch_checkerBoard(d_chirp, M, N);
	checkCudaErrors(cudaGetLastError());


	//Allocate the hologram on the GPU

	Complex* d_recorded_hologram;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram, sizeof(Complex)*M*N));

	Complex* d_stored_frame;
	checkCudaErrors(cudaMalloc((void**)&d_stored_frame, sizeof(Complex)*M*N));

	unsigned char* d_recorded_hologram_uchar;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram_uchar, sizeof(unsigned char)*M*N));

	Complex* d_propagated;
	checkCudaErrors(cudaMalloc((void**)&d_propagated, sizeof(Complex)*M*N));

	float* d_filtered_phase;
	checkCudaErrors(cudaMalloc((void**)&d_filtered_phase, sizeof(float)*M*N));

	float* d_phase_sin;
	checkCudaErrors(cudaMalloc((void**)&d_phase_sin, sizeof(float)*M*N));

	float* d_phase_cos;
	checkCudaErrors(cudaMalloc((void**)&d_phase_cos, sizeof(float)*M*N));

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


	float *vertices = (float *)malloc(M*N * 2 * sizeof(float));

	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			u = (float)j - 0.5f*(float)N;
			v = (float)i - 0.5f*(float)M;
			x = (u) / (0.5f*(float)N);
			y = -(v) / (0.5f*(float)M);

			vertices[k] = x;
			vertices[k + 1] = y;
			k += 2;
		}
	}

	//Load these vertex coordinates into the first vbo
	glBufferData(GL_ARRAY_BUFFER, N*M * 2 * sizeof(GLfloat), vertices, GL_DYNAMIC_DRAW);
	checkGLError(glGetError());

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
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

	//This is the VBO that the complex magnitudes will be written to for visualization.
	glBufferData(GL_ARRAY_BUFFER, M*N * 1 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkGLError(glGetError());

	//Register it as a CUDA graphics resource
	cudaGraphicsResource *cuda_vbo_resource;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo[1], cudaGraphicsMapFlagsWriteDiscard));

	glBindVertexArray(0);

	//Compile vertex and fragment shaders

	//GLuint shaderprogram = initShaders();
	//checkGLError(glGetError());

	// Set up cuFFT stuff
	cufftComplex* d_reconstructed;
	cudaMalloc((void**)&d_reconstructed, sizeof(cufftComplex)*M*N);

	//Set up plan
	cufftResult result = CUFFT_SUCCESS;
	cufftHandle plan;
	result = cufftPlan2d(&plan, M, N, CUFFT_C2C);
	if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }


	bool show_test_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImColor(114, 144, 154);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();
		ImGui::Text("Hello, world!");
		/*
		// 1. Show a simple window
		// Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
		{
		static float f = 0.0f;
		ImGui::Text("Hello, world!");
		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
		ImGui::ColorEdit3("clear color", (float*)&clear_color);
		if (ImGui::Button("Test Window")) show_test_window ^= 1;
		if (ImGui::Button("Another Window")) show_another_window ^= 1;
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		}

		// 2. Show another simple window, this time using an explicit Begin/End pair
		if (show_another_window)
		{
		ImGui::SetNextWindowSize(ImVec2(200,100), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Another Window", &show_another_window);
		ImGui::Text("Hello");
		ImGui::End();
		}

		// 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
		if (show_test_window)
		{
		ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
		ImGui::ShowTestWindow(&show_test_window);
		}
		*/
		// Rendering
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui::Render();
		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplGlfw_Shutdown();
	glfwTerminate();

	return 0;
}
