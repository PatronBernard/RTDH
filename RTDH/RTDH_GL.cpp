#include "RTDH_GL.h"

//Wrapper function to check error codes. Use checkGLError though.
void checkGL(GLenum err, const char* file, const int line);

//Returns a string with the appropriate description for a GL error code.
char* glGetErrorString(GLenum glError);

//Wrapper function to compile a vertex shader
GLuint compileVertexShader();

GLuint initShaders();

//
void assign_vertex_attribute_data(GLuint vbo_pos){
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos); //Bind buffer so we can do stuff with it
	checkGLError(glGetError());
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0); //The array will contain x,y,z,w coordinates.
	checkGLError(glGetError());
	glEnableVertexAttribArray(0);	//We need to enable the attribute index so our shader can use it.
	checkGLError(glGetError());
	glBindBuffer(GL_ARRAY_BUFFER, 0); //Unbind buffer
	checkGLError(glGetError());
}

void checkGL(GLenum err, const char* file, const int line){
	if (err != GL_NO_ERROR){
		fprintf(stderr, "openGL error at %s:%d: %s \n", file, line - 1, glGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

GLuint compileVertexShader(){
	char* source = read_txt("vertex_shader_src.txt");
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &source, 0);
	glCompileShader(vertexShader);
	return vertexShader;
}

char* glGetErrorString(GLenum glError){
	switch (glError){
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

//Read and compile the shaders. 
GLuint initShaders(){
	char* vertex_src = read_txt("vertex_src.glsl");
	char* fragment_src = read_txt("fragment_src.glsl");
	GLint shader_ok;
	int buffersize = 1024;
	char* logbuffer = (char*)malloc(sizeof(char)*buffersize);

	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, (const GLchar**)&vertex_src, 0);
	glCompileShader(vertex_shader);

	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &shader_ok);
	if (!shader_ok) {
		fprintf(stderr, "Failed to compile %s:\n", "vertex_src.glsl");
		glGetShaderInfoLog(vertex_shader, buffersize, NULL, logbuffer);
		fprintf(stderr, "%s \n", logbuffer);
		glDeleteShader(vertex_shader);
		exit(EXIT_FAILURE);
	}

	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, (const GLchar**)&fragment_src, 0);
	glCompileShader(fragment_shader);

	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &shader_ok);
	if (!shader_ok) {
		fprintf(stderr, "Failed to compile %s:\n", "fragment_src.glsl");
		glGetShaderInfoLog(fragment_shader, buffersize, NULL, logbuffer);
		fprintf(stderr, "%s \n", logbuffer);
		glDeleteShader(fragment_shader);
		exit(EXIT_FAILURE);
	}

	GLuint shaderprogram = glCreateProgram();

	glAttachShader(shaderprogram, vertex_shader);
	glAttachShader(shaderprogram, fragment_shader);

	glBindAttribLocation(shaderprogram, 0, "in_Position");
	glBindAttribLocation(shaderprogram, 1, "in_Magnitude");

	glLinkProgram(shaderprogram);
	glUseProgram(shaderprogram);
	return shaderprogram;
}
