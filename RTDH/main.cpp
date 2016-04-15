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

#include "paramgl.h"



#define PI	3.1415926535897932384626433832795028841971693993751058209749
#define PI2 1.570796326794896619231321691639751442098584699687552910487


int main(){
	
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

	int M=(int)height;
	int N=(int)width;
	
	//=========================INITIALIZATION==========================
	
	//Read the reconstruction parameters. 
	reconParameters parameters;
	read_parameters("parameters.txt", &parameters);
	
	//Initialize the GLFW window
	GLFWwindow *window = initGLFW((int)N/4, (int) M/4); 


	//Set a few callbacks
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSetKeyCallback(window, key_callback);

	//Search for CUDA devices and pick the best-suited one. 
	findCUDAGLDevices();

	//Allocate and set up the chirp-function, copy it to the GPU memory. Also checkerboard it 
	// so we don't have to do that in the main loop.
	Complex* h_chirp = (Complex*)malloc(sizeof(Complex)*N*M);
	if (h_chirp == NULL){ printError(); exit(EXIT_FAILURE); }

	construct_chirp(h_chirp, M, N, parameters.lambda, parameters.rec_dist, parameters.pixel_y, parameters.pixel_x);

	Complex* d_chirp;
	checkCudaErrors(cudaMalloc((void**)&d_chirp, sizeof(Complex)*M*N));

	checkCudaErrors(cudaMemcpy(d_chirp, h_chirp, sizeof(Complex)*M*N, cudaMemcpyHostToDevice));
	

	dim3 block(16, 16, 1);
	dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
	launch_checkerBoard(d_chirp, M, N);
	checkCudaErrors(cudaGetLastError());


	//Allocate the hologram on the GPU

	Complex* d_recorded_hologram;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram, sizeof(Complex)*M*N));
	
	Complex* d_stored_frame;
	checkCudaErrors(cudaMalloc((void**)&d_stored_frame, sizeof(Complex)*M*N));

	unsigned char* d_recorded_hologram_uchar;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram_uchar,sizeof(unsigned char)*M*N));

	Complex* d_propagated;
	checkCudaErrors(cudaMalloc((void**)&d_propagated, sizeof(Complex)*M*N));

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

	
	float *vertices = (float *) malloc(M*N * 2 * sizeof(float));
	
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

	//This is the VBO that the complex magnitudes will be written to for visualization.
	glBufferData(GL_ARRAY_BUFFER, M*N * 1 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkGLError(glGetError());

	//Register it as a CUDA graphics resource
	cudaGraphicsResource *cuda_vbo_resource;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo[1], cudaGraphicsMapFlagsWriteDiscard));

	//Compile vertex and fragment shaders

	GLuint shaderprogram = initShaders();
	checkGLError(glGetError());

	// Set up cuFFT stuff
	cufftComplex* d_reconstructed;
	cudaMalloc((void**)&d_reconstructed, sizeof(cufftComplex)*M*N);

	//Set up plan
	cufftResult result = CUFFT_SUCCESS;
	cufftHandle plan;
	result = cufftPlan2d(&plan, M, N, CUFFT_C2C);
	if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
	

	
	//=========================MAIN LOOP==========================
	
	apiController.StartContinuousImageAcquisition(strCameraID);
	AVT::VmbAPI::FramePtr frame;
	VmbUchar_t *image;
	VmbFrameStatusType eReceiveStatus;
	float *vbo_mapped_pointer;
	
	size_t num_bytes;

	// Measure frametime and average it
	double frameTime = 0.0;
	int frameCounter = 0;
	//Number of samples
	int frameLimit = 5;
	//Accumulator
	double totalFrameTime=0.0;

	double averageFrametime = 0.0;
	//std::string wtitle;
	char wtitle[1024];

	//Parameter test
	int numDisplayedSlices;
	ParamListGL *params;
	params = new ParamListGL("test'");
    params->AddParam(new Param<int>("displayed slices", numDisplayedSlices, 0, 256, 1, &numDisplayedSlices));

	//Set up projection matrix 
	//GLuint projection_Handle= glGetUniformLocation(shaderprogram, "Projection");
	//glm::mat4 Projection = glm::ortho(-1.0,1.0,-1.0,1.0);
	//glUniformMatrix4fv(projection_Handle, 1, GL_FALSE, &Projection[0][0]);

	//Start the main loop
	glfwSetTime(0.0);
	while(!glfwWindowShouldClose(window)){		
		//glDisable(GL_DEPTH_TEST);
        //glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        //glEnable(GL_BLEND);
        //params->Render(0, 0);
        //glDisable(GL_BLEND);
        //glEnable(GL_DEPTH_TEST);
		
		//Fetch a frame
		frame=apiController.GetFrame();
		if(	!SP_ISNULL( frame) )
        {      
			frame->GetReceiveStatus(eReceiveStatus);
			//If it is not NULL or incompletem, process it.
			if(eReceiveStatus==VmbFrameStatusComplete){
				//Start measuring time.
				frameTime = glfwGetTime();
				glfwSetTime(0.0);

				frame->GetImage(image);
				//Copy to device
				checkCudaErrors(cudaMemcpy(d_recorded_hologram_uchar,image,
										sizeof(unsigned char)*M*N,
										cudaMemcpyHostToDevice));

				//Convert the image to a complex format.
				launch_unsignedChar2cufftComplex(d_recorded_hologram,
												 d_recorded_hologram_uchar,
												 M,N);

				//If R was pressed, we store this frame. 
				if (storeCurrentFrame){
					storeCurrentFrame=false;
					launch_unsignedChar2cufftComplex(d_stored_frame,
										 d_recorded_hologram_uchar,
										 M,N);
				}

				//For interferometry, we add the stored frame to the current one. 
				if (addRecordedFrameToCurrent==true){
					launch_addComplexPointWiseF(d_stored_frame,d_recorded_hologram, d_recorded_hologram, M,N);
				}

				
				//Map the openGL resource object so we can modify it
				checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer(	(void **)&vbo_mapped_pointer, 
																			&num_bytes, cuda_vbo_resource));

				//Hologram Reconstruction
				if (cMode == cameraModeReconstruct){
						//Multiply with (checkerboarded) chirp function
						launch_matrixMulComplexPointw(d_chirp, d_recorded_hologram, d_propagated,M,N);
						checkCudaErrors(cudaGetLastError());
						//FFT
						result = cufftExecC2C(plan,d_propagated, d_propagated, CUFFT_FORWARD);
						if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
						
						//Write to openGL object	
						if (dMode==displayModeMagnitude){
							launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_propagated,1/(sqrt((float)M*(float)N)), M, N);
						}
						else if (dMode==displayModePhase){
							launch_cufftComplex2PhaseF(vbo_mapped_pointer, d_propagated,0.5/PI, M, N);
							launch_addConstant(vbo_mapped_pointer, 0.5, M, N);

						}
						checkCudaErrors(cudaGetLastError());
				}
				else if(cMode == cameraModeVideo){	
						//Just write the image to the resource
						launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_recorded_hologram, 1.0, M, N);
						checkCudaErrors(cudaGetLastError());	
				}
				else if (cMode == cameraModeFFT){
						//FFT shift
						launch_checkerBoard(d_recorded_hologram,M,N); 

						//FFT
						result = cufftExecC2C(plan,d_recorded_hologram, d_recorded_hologram, CUFFT_FORWARD);
						if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }

						//Write to openGL object	
						if (dMode==displayModeMagnitude){
							launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_recorded_hologram,1/(sqrt((float)M*(float)N)), M, N);
						}
						else if (dMode==displayModePhase){
							launch_cufftComplex2PhaseF(vbo_mapped_pointer, d_recorded_hologram,1./PI, M, N);
						}
						checkCudaErrors(cudaGetLastError());		
				}
				//Note: make this work with every mode? 
				else if (cMode == cameraModeViewStoredFrame){
					//In this case we just display the stored frame.
					launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_stored_frame, 1.0, M, N);
				}

				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));	
				
				//Draw everything
				glDrawArrays(GL_POINTS, 0, (unsigned int)N*(unsigned int)M);
				glfwSwapBuffers(window);
				
				//Check for keypresses
				glfwPollEvents();

				float ratio=width/height;
				

				//Calculate the average frametime
				totalFrameTime+=frameTime;
				frameCounter++;
				if (frameCounter==frameLimit){
					frameCounter=0;
					averageFrametime=totalFrameTime/frameLimit;
					totalFrameTime=0.0;
					sprintf(wtitle,"FPS: %.3f    Frametime: %.5fs",(int)1/averageFrametime,averageFrametime);		
					glfwSetWindowTitle(window, wtitle);							
				}								
				
			}
		}
		//Requeue the frame so we can gather more images
		apiController.QueueFrame(frame);

		checkCudaErrors(cudaThreadSynchronize());
		//
		
		//glfwSetTime(0.0);
		//fps_prev = fps;
		//fps = (int)(0.5*(1. / frameTime + (float)fps_prev));
		//Sleep(1000);
		
		
		//std::cout<< avgFPS << "\n";
		//Update FPS every 15 frames
		//framecounter += 1;
	}


	apiController.StopContinuousImageAcquisition();

	//Export the last reconstructed frame. 
	Complex* h_reconstructed=(Complex*) malloc(sizeof(Complex)*M*N);
	checkCudaErrors(cudaMemcpy(h_reconstructed, d_propagated, sizeof(Complex)*M*N, cudaMemcpyDeviceToHost));

	export_complex_data("reconstructed_hologram.bin", h_reconstructed, M*N);
	

	//Cleanup
	checkCudaErrors(cudaFree(d_recorded_hologram));
	checkCudaErrors(cudaFree(d_recorded_hologram_uchar));
	checkCudaErrors(cudaFree(d_chirp));
	checkCudaErrors(cudaFree(d_propagated));

	free(vertices);
	free(h_chirp);
	free(h_reconstructed);
	
	

	//End GLFW
	glfwTerminate();

	apiController.ShutDown();

	
	fprintf(stderr, "No errors (that I'm aware of)! \n");
	fclose(logfile);
	
	return 0;
}

