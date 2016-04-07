#include "RTDH_GLFW.h"

#include "cameraMode.h"
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
	//Multisampling, not sure if it actually works
	//glfwWindowHint(GLFW_SAMPLES,16);

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
	}
	if (key == GLFW_KEY_1 && action == GLFW_PRESS){
		cMode=cameraModeReconstruct; std::cout << "Reconstruct Mode \n";
	}

	if (key == GLFW_KEY_2 && action == GLFW_PRESS){
		cMode=cameraModeVideo; std::cout << "Video Mode \n";
	}

	if (key == GLFW_KEY_3 && action == GLFW_PRESS){
		cMode=cameraModeFFT; std::cout << "FFT Mode \n";
	}
}

//Allows you to resize the window. 
void window_size_callback(GLFWwindow* window, int width, int height){
	glViewport(0, 0, width, height);
}
