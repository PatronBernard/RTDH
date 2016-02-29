//This header contains various utility functions not relating to CUDA/GLFW/cuFFT

#ifndef RTDH_UTILITY_H
#define RTDH_UTILITY_H

#include <stdio.h>

#include <time.h>

typedef float2 Complex;

#define PI	3.1415926535897932384626433832795028841971693993751058209749

//Allows us to easily print an error to the logfile. 
#define printError() fprintf(stderr, "%s: line %d: %s \n", __FILE__, __LINE__, std::strerror(errno));

// Used for the first lines of the error log
void printTime(FILE* filePtr){
	char text[100];
	time_t now = time(NULL);
	struct tm *t = localtime(&now);
	fprintf(filePtr, "================================================================================ \n");
	strftime(text, sizeof(text) - 1, "%d-%m-%Y (%H:%M:%S)", t);
	fprintf(filePtr, "Error log of %s \n", text);
}

//Parameters struct
struct reconParameters{
	int M;			//Vertical image size
	int N;			//Horizontal image size
	float pixel_x;	//Physical CCD pixel size 
	float pixel_y;  //Physical CCD pixel size
	float lambda;	//Laser wavelength
	float rec_dist;	//Reconstruction distance
};

//Reads a binary file containing 4-byte floats
float* read_data(const char *inputfile){
	//Associate inputfile with a stream via ifptr, open file as binary data.
	FILE *ifPtr;
	ifPtr = fopen(inputfile, "rb");

	if (ifPtr == NULL){
		fprintf(stderr, "read_data: %s", std::strerror(errno));
		exit(EXIT_FAILURE);
	}
	else{
		//Determine file size
		fseek(ifPtr, 0, SEEK_END);
		int ifSize = ftell(ifPtr);
		rewind(ifPtr);
		//Initialize data pointer and allocate a sufficient amount of memory
		float* DataPtr = (float*)malloc(ifSize);

		if (DataPtr == NULL){
			fprintf(stderr, "read_data: %s \n", std::strerror(errno));
			exit(EXIT_FAILURE);
		}

		//Read file.
		else{
			//Read binary data into a memoryblock of size ifSize, pointed to by Data
			int Length = ifSize / (sizeof(float));
			int Elements_Read = fread(DataPtr, 4, Length, ifPtr);
		}
		return DataPtr;
	}
	fclose(ifPtr);
}

//Reads parameters from a text file (each line contains a float and (optionally) a comment.
void read_parameters(const char *inputfile, struct reconParameters *parameters){
	FILE *file_ptr;
	char line[1024];

	file_ptr = fopen(inputfile, "r");
	if (!file_ptr){
		fprintf(stderr, "read_parameters: %s \n", std::strerror(errno));
		exit(EXIT_FAILURE);
		return;
	}
	else{
		fgets(line, 1024, file_ptr);
		if (!feof(file_ptr)){
			parameters->M = (int)std::strtod(line, NULL);
		}
		else{
			fprintf(stderr, "read_parameters: file ended prematurely.\n");
			exit(EXIT_FAILURE);
		}

		fgets(line, 1024, file_ptr);
		if (!feof(file_ptr)){
			parameters->N = (int)std::strtod(line, NULL);
		}
		else{
			fprintf(stderr, "read_parameters: file ended prematurely.\n");
			exit(EXIT_FAILURE);
		}

		fgets(line, 1024, file_ptr);
		if (!feof(file_ptr)){
			parameters->pixel_x = (float)std::strtod(line, NULL);
		}
		else{
			fprintf(stderr, "read_parameters: file ended prematurely.\n");
			exit(EXIT_FAILURE);
		}

		fgets(line, 1024, file_ptr);
		if (!feof(file_ptr)){
			parameters->pixel_y = (float)std::strtod(line, NULL);
		}
		else{
			fprintf(stderr, "read_parameters: file ended prematurely.\n");
			exit(EXIT_FAILURE);
		}

		fgets(line, 1024, file_ptr);
		if (!feof(file_ptr)){
			parameters->lambda = (float)std::strtod(line, NULL);
		}
		else{
			fprintf(stderr, "read_parameters: file ended prematurely.\n");
			exit(EXIT_FAILURE);
		}

		fgets(line, 1024, file_ptr);
		parameters->rec_dist = (float)std::strtod(line, NULL);

		fclose(file_ptr);
	}

}

//Needed for the .glsl files.
char* read_txt(const char* filename){
	FILE* file = fopen(filename, "rb");
	if (file != NULL){
		//Figure out size of file
		fseek(file, 0, SEEK_END);
		long size = ftell(file);
		if (size < 1){
			fprintf(stderr, "read_txt: size is %ld \n", size);
		}
		else{
			fseek(file, 0, SEEK_SET);
		}

		char* file_data = (char*)malloc((size_t)size + 1);

		if (file_data){
			fread(file_data, 1, (size_t)size, file);
			fclose(file);
			file_data[size] = '\0';
			return file_data;
		}
		else{
			fprintf(stderr, "read_txt: Failed to allocate %ld bytes of memory. \n", size);
			exit(EXIT_FAILURE);
			return NULL;
		}

	}
	else{
		fprintf(stderr, "read_txt: Couldn't open: \n %s \n", filename);
		exit(EXIT_FAILURE);
		return NULL;
	}
}

void construct_chirp(Complex* h_chirp, int M, int N, float lambda, float rec_dist, float pixel_x, float pixel_y){
	if (h_chirp != NULL){
		float ch_exp, x, y;

		float a = PI / (rec_dist*lambda);
		
			for (int i = 0; i < M; i++){
				for (int j = 0; j < N; j++){
				x = (float)j - (float)(N / 2.0);
				y = (float)i - (float)(M / 2.0);
				ch_exp = (float)std::pow(x*pixel_x, 2) + (float)std::pow(y*pixel_y, 2);
				h_chirp[i*N + j].x = std::cos(a*ch_exp);
				h_chirp[i*N + j].y = std::sin(a*ch_exp);

			}
		}

		/*
		for (int i = 0; i < M*N; i++){
			//Row and column indices corresponding with linear index i. 
			k = (int)std::floor(float(i / N));
			l = (i % M);

			x = (float)l - (float)(N / 2);
			y = (float)-k + (float)(M / 2);

			ch_exp = (float)std::pow(x*pixel_x, 2) + (float)std::pow(y*pixel_y, 2);

			h_chirp[i].x = std::cos(a*ch_exp);
			h_chirp[i].y = std::sin(a*ch_exp);
		}
		*/
	}
	else{
		fprintf(stderr, "construct_chirp: h_chirp is NULL");
		exit(EXIT_FAILURE);
	}
}

//Export complex data, first the real part, then the imaginary part in 4 byte-floats.
void export_complex_data(const char* inputfile, Complex* data,const int& elements_amount){
	if (data != NULL){
		FILE* OfPtr;
		OfPtr = fopen(inputfile, "wb+");

		if (OfPtr != NULL){
			//Write complex (flattened) array (z_1=x_1+i*y_1 | z_2 = x_2 + i*y_2 | ... ) to 
			//a binary file with the following layout: |x_1 |y_1 |x_2 |y_2 |...
			float* data_flat = (float*)malloc(sizeof(float)*elements_amount * 2);
			if (data_flat!=NULL){
				int n = 0;
				for (int i = 0; i < 2 * elements_amount; i += 2){
					data_flat[i] =  data[n].x;
					data_flat[i + 1] = data[n].y;
					n++;
				}

				int elements_written = fwrite(data_flat, sizeof(float), elements_amount * 2, OfPtr);
				if (elements_written != elements_amount * 2){
					fprintf(stderr, "export_complex_data: only %d out of %d elements written to file. \n", elements_written, elements_amount * 2);
					fclose(OfPtr);
					free(data_flat);
					exit(EXIT_FAILURE);
				}
				free(data_flat);
			}
			else{
				fprintf(stderr, "export_complex_data: %s \n", std::strerror(errno));
				exit(EXIT_FAILURE);
			}
		}
		fclose(OfPtr);
	}
	else{
		fprintf(stderr, "export_data: input data pointer is NULL. \n");
		exit(EXIT_FAILURE);
	}
}
#endif
