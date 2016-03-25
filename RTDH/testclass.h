#ifndef TESTCLASS_H
#define TESTCLASS_H
#define GLEW_STATIC
#include <GL\glew.h>
#include <GLFW\glfw3.h>
class TestClass{
public: 
	TestClass(void);
	~TestClass(void);
	void setWindow(GLFWwindow* window);
	void closeWindow();
private:
	GLFWwindow *window;
};
#endif