#ifndef GL_HELPER_FUNCTIONS_H
#define GL_HELPER_FUNCTIONS_H
#include "RTDH.h"

//Wrapper function to check error codes. Use checkGLError though.
void checkGL(GLenum err, const char* file, const int line);

//Redefine 
#define checkGLError(glError) checkGL((glError),__FILE__,__LINE__);


char* glGetErrorString(GLenum glError);


//Wrapper function to compile a vertex shader
GLuint compileVertexShader();


void checkGL(GLenum err,const char* file, const int line){
	if(err!=GL_NO_ERROR){
		fprintf(stderr,"openGL error at %s:%d: %s \n",file,line-1,glGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

char* glGetErrorString(GLenum glError){
	switch(glError){
	case GL_INVALID_ENUM:
		return "GL_INVALID_ENUM";
	case GL_INVALID_VALUE:
		return "GL_INVALID_VALUE";
	case GL_INVALID_OPERATION:
		return "GL_INVALID_OPERATION";
	case GL_STACK_OVERFLOW:
		return "GL_STACK_OVERFLOW";
	case GL_STACK_UNDERFLOW:
		return "GL_STACK_UNDERFLOW";
	case GL_OUT_OF_MEMORY:
		return "GL_OUT_OF_MEMORY";
	case GL_INVALID_FRAMEBUFFER_OPERATION:
		return "GL_INVALID_FRAMEBUFFER_OPERATION";
	case GL_CONTEXT_LOST:
		return "GL_CONTEXT_LOST";
	default:
		return "UNKNOWN_ERROR";
	}
}

GLuint compileVertexShader(){
	char* source = read_txt("vertex_shader_src.txt");
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &source, 0);
	glCompileShader(vertexShader);
	return vertexShader;
}
#endif