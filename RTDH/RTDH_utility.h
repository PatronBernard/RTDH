//This header contains various utility functions. 

#ifndef RTDH_UTILITY_H
#define RTDH_UTILITY_H

#include <stdio.h>
#include <time.h>

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

#endif
