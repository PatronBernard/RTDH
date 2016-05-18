#include "RTDH_GLFW.h"
#include "globals.h"

//Initialize GLFW, make a window with the right size and initialize GLEW
GLFWwindow* initGLFW(int width, int height){

	//Initialize GLFW
	if (!glfwInit()){
		fprintf(stderr, "Failed to initialize GLFW. \n");
		exit(EXIT_FAILURE);
	}

	//Create GLFW Window
	GLFWwindow* window;
	window = glfwCreateWindow(width, height, "RTDH", NULL, NULL);
	if (!window){
		fprintf(stderr, "Failed to create window. \n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);

	//VSYNC
	glfwSwapInterval(1);

	//GLEW
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	checkGLError(glGetError());
	
	return window;
}

//Callback function that closes the window if escape is pressed.
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
		glfwSetWindowShouldClose(window, GL_TRUE);
		std::cout << "Exiting... \n";
	}
	if (key == GLFW_KEY_2 && action == GLFW_PRESS){
		cMode=cameraModeReconstruct; 
		std::cout << "Fresnell Reconstruction \n";
	}

	if (key == GLFW_KEY_3 && action == GLFW_PRESS){
		cMode=cameraModeReconstructI; 
		std::cout << "Fresnell Reconstruction: Interferometry \n";
	}

	if (key == GLFW_KEY_1 && action == GLFW_PRESS){
		cMode=cameraModeVideo; 
		std::cout << "Video \n";
	}

	if (key == GLFW_KEY_4 && action == GLFW_PRESS){
		cMode=cameraModeFFT; 
		std::cout << "FFT \n";
	}

	if (key == GLFW_KEY_R && action == GLFW_PRESS){
		storeCurrentFrame=true;
		std::cout << "Stored a frame. \n";
	}
	
	if (key == GLFW_KEY_V && action == GLFW_PRESS){
		cMode=cameraModeViewStoredFrame;
		std::cout << "Viewing stored frame. \n";
	}

	if (key == GLFW_KEY_M && action == GLFW_PRESS){
		if ((cMode==cameraModeReconstruct) | (cMode==cameraModeFFT) | (cMode==cameraModeViewStoredFrame))
		{
			dMode=displayModeMagnitude;
			std::cout << "Displaying complex magnitude. \n";
		}	
	}

	if (key == GLFW_KEY_P && action == GLFW_PRESS){
		if ((cMode==cameraModeReconstruct) | (cMode==cameraModeFFT) | (cMode==cameraModeViewStoredFrame))
		{
			dMode=displayModePhase;
			std::cout << "Displaying complex phase. \n";
		}
	}
	if (key == GLFW_KEY_C && action == GLFW_PRESS){
		show_mijn_scherm = !show_mijn_scherm;
	}
	/*
	if (key == GLFW_KEY_I && action == GLFW_PRESS){
		if ((cMode==cameraModeReconstructI) | (cMode==cameraModeFFT))
		{
			if (addRecordedFrameToCurrent==false){
				addRecordedFrameToCurrent=true;
				std::cout << "Adding recorded frame to current frame: ON \n";
			}
			else{
				addRecordedFrameToCurrent=false;
				std::cout << "Adding recorded frame to current frame: OFF \n";
			}
		}
	}
	*/

}

//Allows you to resize the window. 
void window_size_callback(GLFWwindow* window, int width, int height){
	//float aspect_ratio = width / height;
	//int new_height = height;
	//int new_width = (int)aspect_ratio * height;
	//glfwSetWindowSize(window, new_width, new_height);
	glViewport(0, 0, width, height);
}
