//Written by Jan Morez (2016) for a Master Thesis in Physics.
//Supervisor: prof. dr. Joris Dirckx
//Co-supervisor: dr. Sam Van Der Jeught
//Parts of this code will be identical to that of the example code of the libraries used.

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

#include "RTDH_helper_cuda.h"	//CheckCudaErrors
#include "RTDH_utility.h"	
#include "RTDH_GLFW.h"
#include "RTDH_CUDA.h"

//#include "ListFeatures\Source\ListFeatures.h"

//Other
#include <iostream>
#include "globals.h" //Global variables
#include <ctime>

//GLFW
#include <GLFW\glfw3.h>

//Vimba stuff
#include "ApiController.h"
#include "LoadSaveSettings.h"

//imGUI Stuff
#include <imgui.h>
#include "imgui_impl_glfw.h"
#define IM_NEWLINE "\r\n"
static void ShowHotkeys(bool *p_opened, char *hotkeys_text);
static void ShowAbout(bool *showAbout);
void ImGui::ShowStyleEditor(ImGuiStyle* ref);


int main(){
	
	//Redirect stderror to log.txt.
	FILE* logfile = freopen("log.txt", "w", stderr);
	printTime(logfile);
	
	//Initialize the Vimba API and print some info.
	AVT::VmbAPI::Examples::ApiController apiController;
	std::cout << "Vimba Version V " << apiController.GetVersion() << "\n";
	printConsoleInfo();
	//Start the Vimba API
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
	pCamera->Open(VmbAccessModeRead);

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
	GLFWwindow *window = initGLFW((int)N/2, (int) M/2); 
	glfwMakeContextCurrent(window);
	ImGui_ImplGlfw_Init(window, true);
	//glViewport(0, 0, 512, 512);
	

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

	unsigned int* d_recorded_hologram_uchar;
	checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram_uchar,sizeof(unsigned int)*M*N));

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

	glBindVertexArray(0);

	//This is the VBO that the complex magnitudes will be written to for visualization.
	glBufferData(GL_ARRAY_BUFFER, M*N * 1 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkGLError(glGetError());

	glBindVertexArray(0);

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
	//VmbUint16_t *image;
	VmbUchar_t *image;
	VmbFrameStatusType eReceiveStatus;
	float *vbo_mapped_pointer;
	
	size_t num_bytes;

	// Measure frametime and average it
	std::clock_t t;
	t = 0;
	//Time for a signal entire frame
	double frameTime = 0.0;
	//Time to reconstruct
	double recon_time = 0.0;
	double total_recon_time = 0.0;
	int frameCounter = 0;
	//Number of samples
	int frameLimit = 25;
	//Accumulator
	double totalFrameTime=0.0;
	double averageFrametime = 0.0;
	char wtitle[1024];

	//ImGUI Setup

	std::ifstream tfile;
	int length;
	char *hotkeys_text;
	tfile.open("hotkeys.txt");      // open input file
	if (tfile.good()){
		tfile.seekg(0, std::ios::end);    // go to the end
		length = tfile.tellg();           // report location (this is the length)
		tfile.seekg(0, std::ios::beg);    // go back to the beginning
		hotkeys_text = new char[length];    // allocate memory for a buffer of appropriate dimension
		tfile.read(hotkeys_text, length - 1);       // read the whole file into the buffer
		tfile.close();
	}
	else{
		exit(EXIT_FAILURE);
	}

	show_mijn_scherm = true;

	static bool no_titlebar = false;
	static bool no_border = true;
	static bool no_resize = false;
	static bool no_move = false;
	static bool no_scrollbar = false;
	static bool no_collapse = false;
	static bool no_menu = false;
	
	ImGuiWindowFlags window_flags = 0;
	if (no_titlebar)  window_flags |= ImGuiWindowFlags_NoTitleBar;
	if (!no_border)   window_flags |= ImGuiWindowFlags_ShowBorders;
	if (no_resize)    window_flags |= ImGuiWindowFlags_NoResize;
	if (no_move)      window_flags |= ImGuiWindowFlags_NoMove;
	if (no_scrollbar) window_flags |= ImGuiWindowFlags_NoScrollbar;
	if (no_collapse)  window_flags |= ImGuiWindowFlags_NoCollapse;
	if (!no_menu)     window_flags |= ImGuiWindowFlags_MenuBar;

	bool cMode_Fresnell = true;
	bool cMode_Interf = false;
	bool cMode_Video = false;
	bool showHotkeys = false;
	bool p_opened = true;
	bool showAbout = false;
	bool showStyleEditor = false;

	float rec_dist = parameters.rec_dist;
	//These are the rescale and shift parameters to adjust the dynamic range of
	float rescale_param[2] = { 1.0, 0.0 };

	//ImVec4 clear_color = ImColor(114, 144, 154);
	//Start the main loop
	while(!glfwWindowShouldClose(window)){			
		//Fetch a frame
		frame=apiController.GetFrame();
		if(	!SP_ISNULL( frame) )
        {      
			frame->GetReceiveStatus(eReceiveStatus);
			//If it is not NULL or incomplete, process it.
			if(eReceiveStatus==VmbFrameStatusComplete){
				//Start measuring time.
				frameTime = (double) (clock()-t)/CLOCKS_PER_SEC;
				t = clock();
				frame->GetImage(image);
				//Copy to device
				checkCudaErrors(cudaMemcpy(d_recorded_hologram_uchar,image,
										sizeof(unsigned char)*M*N,
										cudaMemcpyHostToDevice));

				//Convert the image to a complex format.
				launch_unsignedChar2cufftComplex(d_recorded_hologram,
												 d_recorded_hologram_uchar,
												 M,N);
		
				//Map the openGL resource object so we can modify it
				checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer(	(void **)&vbo_mapped_pointer, 
																			&num_bytes, cuda_vbo_resource));
				//This should be in an if-clause so that it only recalculates when it actually changed.
				launch_constructChirp(d_chirp, rec_dist, parameters.lambda, parameters.pixel_x, parameters.pixel_y, M, N);
				launch_checkerBoard(d_chirp, M, N);

				//Ordinary Hologram Reconstruction
				if (cMode == cameraModeReconstruct){
					//Multiply with (checkerboarded) chirp function
					launch_matrixMulComplexPointw(d_chirp, d_recorded_hologram, d_propagated,M,N);
					checkCudaErrors(cudaGetLastError());
					//FFT
					result = cufftExecC2C(plan,d_propagated, d_propagated, CUFFT_FORWARD);
					if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
					
					//Write to openGL object	
					if (dMode==displayModeMagnitude){
						launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_propagated,5.0/(sqrt((float)M*(float)N)), M, N);
						launch_rescaleAndShiftF(vbo_mapped_pointer, rescale_param[0], rescale_param[1], M, N);
					}
					else if (dMode==displayModePhase){
						launch_cufftComplex2PhaseF(vbo_mapped_pointer, d_propagated,0.5/PI, M, N);
						launch_addConstant(vbo_mapped_pointer, 0.5, M, N);
					}
					checkCudaErrors(cudaGetLastError());
				}
				else if(cMode == cameraModeReconstructI){
					//Multiply with (checkerboarded) chirp function
					launch_matrixMulComplexPointw(d_chirp, d_recorded_hologram, d_propagated,M,N);
					checkCudaErrors(cudaGetLastError());
					//FFT
					result = cufftExecC2C(plan,d_propagated, d_propagated, CUFFT_FORWARD);
					if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
					
					//Calculate the phase 1difference and display it.
					launch_phaseDifference(d_stored_frame, d_propagated, d_filtered_phase, 1.0, 0.0, M, N);
					//checkCudaErrors(cudaGetLastError());
					launch_sin(d_filtered_phase, d_phase_sin, M, N);
					//checkCudaErrors(cudaGetLastError());
					launch_cos(d_filtered_phase, d_phase_cos, M, N);
					//checkCudaErrors(cudaGetLastError());
					launch_filterPhase(d_phase_sin, d_phase_cos, vbo_mapped_pointer, 5, M, N);
					//checkCudaErrors(cudaGetLastError());
					launch_rescaleAndShiftF(vbo_mapped_pointer, 0.5 / PI, 0.5, M, N);
				}
				else if(cMode == cameraModeVideo){	
						//Just write the image to the resource
						launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_recorded_hologram, 1.0, M, N);
						checkCudaErrors(cudaGetLastError());	
						launch_rescaleAndShiftF(vbo_mapped_pointer, rescale_param[0], rescale_param[1], M, N);
				}
				else if (cMode == cameraModeFFT){
						//FFT shift
						launch_checkerBoard(d_recorded_hologram,M,N); 
						checkCudaErrors(cudaGetLastError());
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
					if (dMode==displayModeMagnitude){
							launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_stored_frame,1/(sqrt((float)M*(float)N)), M, N);
						}
						else if (dMode==displayModePhase){
							launch_cufftComplex2PhaseF(vbo_mapped_pointer, d_stored_frame,0.5/PI, M, N);
							launch_addConstant(vbo_mapped_pointer,0.5,M,N);
						}
				}
				//Measure the time to reconstruct a frame
				recon_time= (double) (clock()-t)/CLOCKS_PER_SEC;
				total_recon_time += recon_time;
				

				//If R was pressed, we store the last reconstructed frame.
				if (storeCurrentFrame){
					storeCurrentFrame=false;
					cudaMemcpy(d_stored_frame,d_propagated,sizeof(Complex)*M*N,cudaMemcpyDeviceToDevice);
				}

				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));	

				//Calculate the average frametime
				totalFrameTime+=frameTime;
				frameCounter++;
				if (frameCounter==frameLimit){
					frameCounter=0;
					averageFrametime=totalFrameTime / (float)frameLimit;
					totalFrameTime=0.0;
					
					sprintf(wtitle,"FPS: %.3f    Frametime: %.5fs    Reconstruction time: %.5fs",
						(int)1 / averageFrametime, averageFrametime, total_recon_time / (float)frameLimit);
					total_recon_time = 0.0;
					glfwSetWindowTitle(window, wtitle);							
				}								
				
			}
		}

		glfwPollEvents();
		//Draw the interface, not to be confused with the image frame! This is ImGui's frame object!
		ImGui_ImplGlfw_NewFrame();
		if (show_mijn_scherm){
			if (showHotkeys){
				ShowHotkeys(&showHotkeys, hotkeys_text);
			}
			if (showAbout){
				ShowAbout(&showAbout);
			}
			if (showStyleEditor){
				ImGui::SetNextWindowPos(ImVec2(460, 520), ImGuiSetCond_FirstUseEver);
				ImGui::Begin("Style editor", &showStyleEditor, 0);
				ImGui::ShowStyleEditor(NULL);
				ImGui::End();
			}

			ImGui::SetNextWindowPos(ImVec2(400, 450), ImGuiSetCond_FirstUseEver);
			ImGui::Begin("Controls (C)", &show_mijn_scherm, window_flags);
			if (ImGui::BeginMenuBar())
			{
				if (ImGui::BeginMenu("File"))
				{	
					//if (ImGui::MenuItem("Look for camera...", "F5")) {}
					//if (ImGui::MenuItem("Save", "Ctrl+S")) {}
					//if (ImGui::MenuItem("Save As..", "Ctrl+Shift+S")) {}
					if (ImGui::MenuItem("Quit", "Esc")) {
						break;
					}
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Mode")){
					static int e = 0;
					if (ImGui::MenuItem("Video", "1", cMode == 0, true)){
						cMode = cameraModeVideo;
					};

					if (ImGui::MenuItem("Fresnel Reconstruction (Magnitude)", "2", cMode == 1, true)){
						cMode = cameraModeReconstruct;
					};

					if (ImGui::MenuItem("Holographic Interferometry(Phase)", "3", cMode == 2, true)){
						cMode = cameraModeReconstructI;
					};

					if (ImGui::MenuItem("FFT", "4", cMode == 3, true)){
						cMode = cameraModeFFT;
					};

					if (ImGui::MenuItem("View currently stored frame.", "V", cMode == 4, true)){
						cMode = cameraModeViewStoredFrame;
					};
					ImGui::EndMenu();
				}
				//This was originally so the user can pick the colormap. Shouldn't be too hard to implement.
				/*
				if (ImGui::BeginMenu("Settings")){
					if (ImGui::BeginMenu("Colormap")){
						if (ImGui::MenuItem("Hot", NULL, cMap == colormapHot, true)){
							cMap = colormapHot;
						};
						if (ImGui::MenuItem("Jet", NULL, cMap == colormapJet, true)){
							cMap = colormapJet;
						};
						if (ImGui::MenuItem("Green", NULL, cMap == colormapGreen, true)){
							cMap = colormapGreen;
						};
						ImGui::EndMenu();
					}
					if (ImGui::MenuItem("Style editor", NULL, false, true)){
						showStyleEditor ^= 1;
					}
					ImGui::EndMenu();
				}
				*/
				if (ImGui::BeginMenu("Help")){
					if (ImGui::MenuItem("Show Hotkeys", NULL, false, true)){
						showHotkeys ^= 1;
					};
					if (ImGui::MenuItem("About", NULL, false, true)){
						showAbout ^= 1;
					}
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			ImGui::TextWrapped("Recontruction distance(m)");
			ImGui::SliderFloat("", &rec_dist, -1.0, 1.0);
			ImGui::TextWrapped("Use these linear scaling factors (respectively a and b so that Y=aX+b) to adjust the brightness & contrast of the image. Ideally you want your dynamic range of interest to be mapped to the interval [0,1]");
			ImGui::InputFloat2("", rescale_param);
			ImGui::End();
		}

		//Draw points
		glUseProgram(shaderprogram);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, (unsigned int)N*(unsigned int)M);
		glBindVertexArray(0);

		//Needed to draw the interface correctly (see ImGui documentation).
		glUseProgram(0);
		ImGui::Render();
		glfwSwapBuffers(window);

		//Requeue the frame so we can gather more images
		apiController.QueueFrame(frame);
		checkCudaErrors(cudaThreadSynchronize());
		
	}

	//Stop acquiring images
	apiController.StopContinuousImageAcquisition();

	//Export the stored frame
	Complex* h_reconstructed=(Complex*) malloc(sizeof(Complex)*M*N);
	checkCudaErrors(cudaMemcpy(h_reconstructed, d_stored_frame, sizeof(Complex)*M*N, cudaMemcpyDeviceToHost));
	export_complex_data("reconstructed_hologram.bin", h_reconstructed, M*N);
	
	//Cleanup
	checkCudaErrors(cudaFree(d_recorded_hologram));
	checkCudaErrors(cudaFree(d_recorded_hologram_uchar));
	checkCudaErrors(cudaFree(d_chirp));
	checkCudaErrors(cudaFree(d_propagated));
	checkCudaErrors(cudaFree(d_stored_frame));
	checkCudaErrors(cudaFree(d_filtered_phase));
	checkCudaErrors(cudaFree(d_phase_cos));
	checkCudaErrors(cudaFree(d_phase_sin));

	free(vertices);
	free(h_chirp);
	free(h_reconstructed);
	
	//Shut down the GUI?
	ImGui_ImplGlfw_Shutdown();

	//End GLFW
	glfwTerminate();

	apiController.ShutDown();

	
	fprintf(stderr, "No errors (that I'm aware of)! \n");
	fclose(logfile);
	
	return 0;
}

static void ShowHotkeys(bool* p_opened, char  *hotkeys_text){
	ImGui::SetNextWindowPos(ImVec2(300, 200), ImGuiSetCond_FirstUseEver);
	if (ImGui::Begin("Hotkeys", p_opened, NULL)){
		ImGui::TextWrapped(hotkeys_text);
		ImGui::End();
	}

}

static void ShowAbout(bool *showAbout){
	ImGui::SetNextWindowPos(ImVec2(300, 200), ImGuiSetCond_FirstUseEver);
	if (ImGui::Begin("About", showAbout, NULL)){
		ImGui::TextWrapped("Written in 2016 by Jan Morez for a Master Thesis in Physics, University of Antwerp. \nSource code can be found here: \nhttps://github.com/PatronBernard/RTDH");
		ImGui::End();
	}
};

void ImGui::ShowStyleEditor(ImGuiStyle* ref)
{
	ImGuiStyle& style = ImGui::GetStyle();

	const ImGuiStyle def; // Default style
	if (ImGui::Button("Revert Style"))
		style = ref ? *ref : def;
	if (ref)
	{
		ImGui::SameLine();
		if (ImGui::Button("Save Style"))
			*ref = style;
	}

	ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.55f);

	if (ImGui::TreeNode("Rendering"))
	{
		ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);
		ImGui::Checkbox("Anti-aliased shapes", &style.AntiAliasedShapes);
		ImGui::PushItemWidth(100);
		ImGui::DragFloat("Curve Tessellation Tolerance", &style.CurveTessellationTol, 0.02f, 0.10f, FLT_MAX, NULL, 2.0f);
		if (style.CurveTessellationTol < 0.0f) style.CurveTessellationTol = 0.10f;
		ImGui::DragFloat("Global Alpha", &style.Alpha, 0.005f, 0.20f, 1.0f, "%.2f"); // Not exposing zero here so user doesn't "lose" the UI (zero alpha clips all widgets). But application code could have a toggle to switch between zero and non-zero.
		ImGui::PopItemWidth();
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Sizes"))
	{
		ImGui::SliderFloat2("WindowPadding", (float*)&style.WindowPadding, 0.0f, 20.0f, "%.0f");
		ImGui::SliderFloat("WindowRounding", &style.WindowRounding, 0.0f, 16.0f, "%.0f");
		ImGui::SliderFloat("ChildWindowRounding", &style.ChildWindowRounding, 0.0f, 16.0f, "%.0f");
		ImGui::SliderFloat2("FramePadding", (float*)&style.FramePadding, 0.0f, 20.0f, "%.0f");
		ImGui::SliderFloat("FrameRounding", &style.FrameRounding, 0.0f, 16.0f, "%.0f");
		ImGui::SliderFloat2("ItemSpacing", (float*)&style.ItemSpacing, 0.0f, 20.0f, "%.0f");
		ImGui::SliderFloat2("ItemInnerSpacing", (float*)&style.ItemInnerSpacing, 0.0f, 20.0f, "%.0f");
		ImGui::SliderFloat2("TouchExtraPadding", (float*)&style.TouchExtraPadding, 0.0f, 10.0f, "%.0f");
		ImGui::SliderFloat("IndentSpacing", &style.IndentSpacing, 0.0f, 30.0f, "%.0f");
		ImGui::SliderFloat("ScrollbarSize", &style.ScrollbarSize, 1.0f, 20.0f, "%.0f");
		ImGui::SliderFloat("ScrollbarRounding", &style.ScrollbarRounding, 0.0f, 16.0f, "%.0f");
		ImGui::SliderFloat("GrabMinSize", &style.GrabMinSize, 1.0f, 20.0f, "%.0f");
		ImGui::SliderFloat("GrabRounding", &style.GrabRounding, 0.0f, 16.0f, "%.0f");
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Colors"))
	{
		static int output_dest = 0;
		static bool output_only_modified = false;
		if (ImGui::Button("Copy Colors"))
		{
			if (output_dest == 0)
				ImGui::LogToClipboard();
			else
				ImGui::LogToTTY();
			ImGui::LogText("ImGuiStyle& style = ImGui::GetStyle();" IM_NEWLINE);
			for (int i = 0; i < ImGuiCol_COUNT; i++)
			{
				const ImVec4& col = style.Colors[i];
				const char* name = ImGui::GetStyleColName(i);
				if (!output_only_modified || memcmp(&col, (ref ? &ref->Colors[i] : &def.Colors[i]), sizeof(ImVec4)) != 0)
					ImGui::LogText("style.Colors[ImGuiCol_%s]%*s= ImVec4(%.2ff, %.2ff, %.2ff, %.2ff);" IM_NEWLINE, name, 22 - (int)strlen(name), "", col.x, col.y, col.z, col.w);
			}
			ImGui::LogFinish();
		}
		ImGui::SameLine(); ImGui::PushItemWidth(120); ImGui::Combo("##output_type", &output_dest, "To Clipboard\0To TTY"); ImGui::PopItemWidth();
		ImGui::SameLine(); ImGui::Checkbox("Only Modified Fields", &output_only_modified);

		static ImGuiColorEditMode edit_mode = ImGuiColorEditMode_RGB;
		ImGui::RadioButton("RGB", &edit_mode, ImGuiColorEditMode_RGB);
		ImGui::SameLine();
		ImGui::RadioButton("HSV", &edit_mode, ImGuiColorEditMode_HSV);
		ImGui::SameLine();
		ImGui::RadioButton("HEX", &edit_mode, ImGuiColorEditMode_HEX);
		//ImGui::Text("Tip: Click on colored square to change edit mode.");

		static ImGuiTextFilter filter;
		filter.Draw("Filter colors", 200);

		ImGui::BeginChild("#colors", ImVec2(0, 300), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
		ImGui::PushItemWidth(-160);
		ImGui::ColorEditMode(edit_mode);
		for (int i = 0; i < ImGuiCol_COUNT; i++)
		{
			const char* name = ImGui::GetStyleColName(i);
			if (!filter.PassFilter(name))
				continue;
			ImGui::PushID(i);
			ImGui::ColorEdit4(name, (float*)&style.Colors[i], true);
			if (memcmp(&style.Colors[i], (ref ? &ref->Colors[i] : &def.Colors[i]), sizeof(ImVec4)) != 0)
			{
				ImGui::SameLine(); if (ImGui::Button("Revert")) style.Colors[i] = ref ? ref->Colors[i] : def.Colors[i];
				if (ref) { ImGui::SameLine(); if (ImGui::Button("Save")) ref->Colors[i] = style.Colors[i]; }
			}
			ImGui::PopID();
		}
		ImGui::PopItemWidth();
		ImGui::EndChild();

		ImGui::TreePop();
	}

	ImGui::PopItemWidth();
};