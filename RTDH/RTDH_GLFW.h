#ifndef RTDH_GLFW_H
#define RTDH_GLFW_H

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "RTDH_GL.h"
#include <stdio.h>
#include <stdlib.h>



//Initialize GLFW, make a window with the right size and initialize GLEW
GLFWwindow* initGLFW(int width, int height);

//Callback function that closes the window if escape is pressed.
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

//Allows you to resize the window. 
void window_size_callback(GLFWwindow* window, int width, int height);

#endif