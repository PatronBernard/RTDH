#include <iostream>
#include "FrameObserverRTDH.h"
#include "kernels.h"
//GLFW
#include <GLFW\glfw3.h>
#include "RTDH_utility.h"
#include "RTDH_CUDA.h"

namespace AVT {
namespace VmbAPI {
FrameObserverRTDH::FrameObserverRTDH( CameraPtr pCamera)
    :   IFrameObserver( pCamera )
{
}

void FrameObserverRTDH::FrameReceived( const FramePtr pFrame)
{
    if(! SP_ISNULL( pFrame ))
    {
        VmbFrameStatusType status;
        VmbErrorType Result;
        Result = SP_ACCESS( pFrame)->GetReceiveStatus( status);
        if( VmbErrorSuccess == Result && VmbFrameStatusComplete == status)
        {
			 std::cout<< "Received a frame! \n";
			pFrame->GetImage(pImage);

			VmbErrorType vmb_err= VmbErrorSuccess;
			VmbUint32_t M;
			pFrame->GetHeight(M);

			VmbUint32_t N;
			pFrame->GetWidth(N);
			//Start measuring frame time
			glfwSetTime(0.0);
		
			//Set up the grid
			dim3 block(16, 16, 1);
			//I added the +1 because it might round down which can mean that not all pixels are processed in each kernel. 
			dim3 grid((unsigned int)M / block.x+1, (unsigned int)N / block.y+1, 1);
		
			//Fetch an image, copy it to the device and convert it

			SP_ACCESS( pFrame )->GetBuffer( pImage );

			std::cout << (int) pImage[150] << "\n";
			//unsigned char* d_recorded_hologram_uchar;
			//checkCudaErrors(cudaMalloc((void**)&d_recorded_hologram_uchar,sizeof(unsigned char)*(unsigned int)M*(unsigned int)N));
			checkCudaErrors(cudaMemcpy(	d_recorded_hologram_uchar,pImage,
									sizeof(unsigned char)*(unsigned int)M*(unsigned int)N,
									cudaMemcpyHostToDevice));
			
			launch_unsignedChar2cufftComplex(this->d_recorded_hologram,
											 this->d_recorded_hologram_uchar,
											 (unsigned int)M,(unsigned int)N);
			/*
			matrixMulComplexPointw <<<grid, block >>>(d_chirp, d_recorded_hologram, d_propagated, parameters.M, parameters.N);
			checkCudaErrors(cudaGetLastError());

			result = cufftExecC2C(plan,d_propagated, d_propagated, CUFFT_FORWARD);
			if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
		

			
			float *vbo_mapped_pointer; //This is the pointer that we'll write the result to for display in OpenGL.
			checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));

			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vbo_mapped_pointer, &num_bytes, this->cuda_vbo_resource));

			launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, d_recorded_hologram, (unsigned int)M, (unsigned int)N);
			
			//restrictToRange << <grid, block >> >(dptr, parameters.M, parameters.N);

			checkCudaErrors(cudaGetLastError());
		
			// unmap buffer object
			checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
		
			int w_width, w_height;
			glfwGetWindowSize(window, &w_width, &w_height);
		
			//glm::mat4 Projection = glm::ortho(-(float)parameters.N / (float)w_width, (float)parameters.N / (float)w_width, -(float)parameters.M / (float)w_height, (float)parameters.M / (float)w_height);
			glm::mat4 Projection = glm::ortho(-1.0,1.0,-1.0,1.0);
			
			glUniformMatrix4fv(projection_Handle, 1, GL_FALSE, &Projection[0][0]);
			*/

			glDrawArrays(GL_POINTS, 0, (unsigned int)N*(unsigned int)M);

			glfwSwapBuffers(window);
			glfwPollEvents();

			
		
        }
        else
        {
            std::cout<<"frame incomplete\n";
        }
    }
    else
    {
        std::cout <<" frame pointer NULL\n";
    }

    m_pCamera->QueueFrame( pFrame );
}

void FrameObserverRTDH::loadAllTheOtherStuff(GLFWwindow* window, 
				GLuint shaderprogram, 
				GLuint projection_Handle, 
				cudaGraphicsResource *cuda_vbo_resource, 
				Complex* d_recorded_hologram, 
				Complex* d_chirp,
				Complex* d_propagated,
				cufftHandle plan,
				std::string strCameraID,
				unsigned char* d_recorded_hologram_uchar){
	this->window=window;
	this->shaderprogram=shaderprogram;
	this->projection_Handle=projection_Handle;
	this->cuda_vbo_resource=cuda_vbo_resource;
	this->d_recorded_hologram=d_recorded_hologram;
	this->d_chirp=d_chirp;
	this->d_propagated=d_propagated;
	this->plan=plan;
	this->strCameraID=strCameraID;
	this->d_recorded_hologram_uchar=d_recorded_hologram_uchar;
}
}}