#ifndef RTDH_GL_H
#define RTDH_GL_H
#define GLEW_STATIC
#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include "RTDH_utility.h"

//Wrapper function to check error codes. Use checkGLError though.
void checkGL(GLenum err, const char* file, const int line);

//Redefine 
#define checkGLError(glError) checkGL((glError),__FILE__,__LINE__);

//Returns a string with the appropriate description for a GL error code.
char* glGetErrorString(GLenum glError);

//Wrapper function to compile a vertex shader
GLuint compileVertexShader();

GLuint initShaders();

//
void assign_vertex_attribute_data(GLuint vbo_pos);

void checkGL(GLenum err, const char* file, const int line);

GLuint compileVertexShader();

char* glGetErrorString(GLenum glError);
//Read and compile the shaders. 
GLuint initShaders();
#endif