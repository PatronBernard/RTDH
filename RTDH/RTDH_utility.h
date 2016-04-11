//This header contains various utility functions not relating to CUDA/GLFW/cuFFT

#ifndef RTDH_UTILITY_H
#define RTDH_UTILITY_H

#include "ApiController.h"
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <cufftXt.h>

typedef float2 Complex;

#define PI	3.1415926535897932384626433832795028841971693993751058209749

//Parameters struct
struct reconParameters{
	float pixel_x;	//Physical CCD pixel size 
	float pixel_y;  //Physical CCD pixel size
	float lambda;	//Laser wavelength
	float rec_dist;	//Reconstruction distance
};


// Used for the first lines of the error log
void printTime(FILE* filePtr);

//Allows us to easily print an error to the logfile. 
#define printError() fprintf(stderr, "%s: line %d: %s \n", __FILE__, __LINE__, std::strerror(errno));

//Helper function that returns an error string for Vimba API errors
char* getVimbaErrorStr(VmbErrorType vmb_err);
void printVimbaErr(VmbErrorType vmb_err,const char *const file,const int line);

//Prints the file and line number for Vimba errors.
#define printVimbaError(vmb_err) printVimbaErr(vmb_err, __FILE__, __LINE__);

//Reads a binary file containing 4-byte floats
float* read_data(const char *inputfile);

//Reads parameters from a text file (each line contains a float and (optionally) a comment.
void read_parameters(const char *inputfile, struct reconParameters *parameters);

//Needed for the .glsl files.
char* read_txt(const char* filename);

void construct_chirp(Complex* h_chirp, int M, int N, float lambda, float rec_dist, float pixel_x, float pixel_y);

//Export complex data, first the real part, then the imaginary part in 4 byte-floats.
void export_complex_data(const char* inputfile, Complex* data,const int& elements_amount);

void computeFPS();

void printConsoleInfo();
#endif