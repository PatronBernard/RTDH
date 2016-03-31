/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        FrameObserver.h

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

#ifndef AVT_VMBAPI_EXAMPLES_FRAMEOBSERVER
#define AVT_VMBAPI_EXAMPLES_FRAMEOBSERVER

#include <queue>
#include <VimbaCPP/Include/VimbaCPP.h>
#include <GL\glew.h>
#include <cufftXt.h>

typedef float2 Complex;
#include "ProgramConfig.h"

#include <GLFW\glfw3.h>



namespace AVT {
namespace VmbAPI {
namespace Examples {

#define WM_FRAME_READY WM_USER + 1

class FrameObserverRTDH : virtual public IFrameObserver
{
  public:
    // We pass the camera that will deliver the frames to the constructor
      FrameObserverRTDH( CameraPtr pCamera ) : IFrameObserver( pCamera ) {;}
    
    // This is our callback routine that will be executed on every received frame
    virtual void FrameReceived( const FramePtr pFrame );

	//Passes all the resources so that FrameReceived can reconstruct the hologram
	void loadAllTheOtherStuff(GLFWwindow* window, 
				GLuint shaderprogram, 
				GLuint projection_Handle, 
				cudaGraphicsResource *cuda_vbo_resource, 
				Complex* d_recorded_hologram, 
				Complex* d_chirp,
				Complex* d_propagated,
				cufftHandle plan,
				std::string strCameraID,
				unsigned char* d_recorded_hologram_uchar);


    // After the view has been notified about a new frame it can pick it up
    FramePtr GetFrame();

	void Reconstruct();

    // Clears the double buffer frame queue
    void ClearFrameQueue();

  private:
    // Since a MFC message cannot contain a whole frame
    // the frame observer stores all FramePtr
    std::queue<FramePtr> m_Frames;
    AVT::VmbAPI::Mutex m_FramesMutex;

	//Things needed for RTDH
	GLFWwindow* window;
	GLuint shaderprogram;
	GLuint projection_Handle;
	cudaGraphicsResource *cuda_vbo_resource;
	Complex* d_recorded_hologram;
	Complex* d_chirp;
	Complex* d_propagated;
	cufftHandle plan;
	std::string strCameraID;
	AVT::VmbAPI::FramePtr pFrame;
	unsigned char* d_recorded_hologram_uchar;
};

}}} // namespace AVT::VmbAPI::Examples

#endif
