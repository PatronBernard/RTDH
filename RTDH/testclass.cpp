#include "testclass.h"

TestClass::TestClass(void){
}

TestClass::~TestClass(void){
}

void TestClass::setWindow(GLFWwindow* window){
	this->window=window;
}

void TestClass::closeWindow(){
	glfwSetWindowSize(this->window,1024,512);

}