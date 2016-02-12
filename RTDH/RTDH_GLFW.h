#ifndef RTDH_GLFW_H
#define RTDH_GLFW_H

#include <GL/glew.h>
#include <GLFW\glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include "RTDH_GL.h"


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
#endif