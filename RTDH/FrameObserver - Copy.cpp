/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        FrameObserver.cpp

  Description: The frame observer that is used for notifications from VimbaCPP
               regarding the arrival of a newly acquired frame.

-------------------------------------------------------------------------------

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/


#include "FrameObserver.h"
#include "RTDH_CUDA.h"
#include "RTDH_utility.h"
#include "kernels.h"
#include <stdio.h>

namespace AVT {
namespace VmbAPI {
namespace Examples {

void FrameObserverRTDH::FrameReceived( const FramePtr pFrame )
{
    bool bQueueDirectly = true;
    VmbFrameStatusType eReceiveStatus;
	
    if( VmbErrorSuccess == pFrame->GetReceiveStatus( eReceiveStatus ) )
    {

                // Lock the frame queue
                m_FramesMutex.Lock();
                // We store the FramePtr
                
				m_Frames.push( pFrame );
				
                // Unlock frame queue
                m_FramesMutex.Unlock();

				this->Reconstruct();
			          
				bQueueDirectly = false;

    }

    // If any error occurred we queue the frame without notification
    if( true == bQueueDirectly )
    {
        m_pCamera->QueueFrame( pFrame );
    }
}

void FrameObserverRTDH::Reconstruct(){
	//Fetch the frame from the frame queue
	/*
	AVT::VmbAPI::FramePtr currentFrame = this->GetFrame();

	//Get the size
	VmbUint32_t M;
	VmbErrorType vmb_err=currentFrame->GetHeight(M);
	if(vmb_err != VmbErrorSuccess){
		printVimbaError(vmb_err); exit(EXIT_FAILURE);}
	VmbUint32_t N;
	vmb_err=currentFrame->GetWidth(N);
	if(vmb_err != VmbErrorSuccess){
		printVimbaError(vmb_err); exit(EXIT_FAILURE);}
		*/
	printf("Received a frame! \n");
	//VmbUchar_t *image;
	//currentFrame->GetImage(image);
	/*
	checkCudaErrors(cudaMemcpy(this->d_recorded_hologram_uchar,image,
									sizeof(unsigned char)*(unsigned int)M*(unsigned int)N,
									cudaMemcpyHostToDevice));
	
	launch_Modify(d_recorded_hologram_uchar,M, N);

	unsigned char *resultaat= (unsigned char*) malloc(sizeof(unsigned char)*(unsigned int)M*(unsigned int)N);



	checkCudaErrors(cudaMemcpy(resultaat, this->d_recorded_hologram_uchar,
									sizeof(unsigned char)*(unsigned int)M*(unsigned int)N,
									cudaMemcpyDeviceToHost));
	
	printf("%u ========== %u \n",image[150],resultaat[150]);
	*/

	/*
	launch_unsignedChar2cufftComplex(this->d_recorded_hologram,
											 this->d_recorded_hologram_uchar,
											 (unsigned int)M,(unsigned int)N);
			
	launch_matrixMulComplexPointw(this->d_chirp, this->d_recorded_hologram, this->d_propagated,(unsigned int)M, (unsigned int)N);
		checkCudaErrors(cudaGetLastError());

	cufftResult_t result = cufftExecC2C(this->plan,this->d_propagated, this->d_propagated, CUFFT_FORWARD);
		if (result != CUFFT_SUCCESS) { printCufftError(); exit(EXIT_FAILURE); }
		*/

		/*
		float *vbo_mapped_pointer; //This is the pointer that we'll write the result to for display in OpenGL.
		checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
			
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vbo_mapped_pointer, &num_bytes, this->cuda_vbo_resource));

		launch_cufftComplex2MagnitudeF(vbo_mapped_pointer, this->d_recorded_hologram, (unsigned int)M, (unsigned int)N);
			
		//restrictToRange << <grid, block >> >(dptr, parameters.M, parameters.N);

		checkCudaErrors(cudaGetLastError());
		
			
		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, 0));
			
		//int w_width, w_height;
		//glfwGetWindowSize(this->window, &w_width, &w_height);
		
		//glm::mat4 Projection = glm::ortho(-(float)parameters.N / (float)w_width, (float)parameters.N / (float)w_width, -(float)parameters.M / (float)w_height, (float)parameters.M / (float)w_height);
		//glm::mat4 Projection = glm::ortho(-1.0,1.0,-1.0,1.0);
			
		//glUniformMatrix4fv(projection_Handle, 1, GL_FALSE, &Projection[0][0]);
		*/

	//glDrawArrays(GL_POINTS, 0, (unsigned int)N*(unsigned int)M);

	glfwSwapBuffers(this->window);
	glfwPollEvents();


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
	//this->window=window;
					//Initialize the GLFW window
	//GLFWwindow *window = initGLFW(parameters.N, parameters.M); 


	this->window = window;
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

// Returns the oldest frame that has not been picked up yet
FramePtr FrameObserverRTDH::GetFrame()
{
    // Lock frame queue
    m_FramesMutex.Lock();
    // Pop the frame from the queue
    FramePtr res;
    if( ! m_Frames.empty() )
    {
        res = m_Frames.front();
        m_Frames.pop();
    }
    // Unlock the frame queue
    m_FramesMutex.Unlock();
    return res;
}

void FrameObserverRTDH::ClearFrameQueue()
{
    // Lock the frame queue
    m_FramesMutex.Lock();
    // Clear the frame queue and release the memory
    std::queue<FramePtr> empty;
    std::swap( m_Frames, empty );
    // Unlock the frame queue
    m_FramesMutex.Unlock();
}

}}} // namespace AVT::VmbAPI::Examples
