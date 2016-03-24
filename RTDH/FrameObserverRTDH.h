
#ifndef _FRAMEOBSERVERRTDH_H
#define _FRAMEOBSERVERRTDH_H

#include <GL\glew.h>
#include <cufftXt.h>
//#include "RTDH_utility.h"
typedef float2 Complex;

#include <queue>
#include "VimbaCPP/Include/VimbaCPP.h"
#include "ProgramConfig.h"

#include <GLFW\glfw3.h>

//Idea: derive this class from IFrame Observer
namespace AVT {
namespace VmbAPI {
class FrameObserverRTDH: virtual public AVT::VmbAPI::IFrameObserver
{	
public:
	// We pass the camera that will deliver the frames to the constructor
	FrameObserverRTDH( CameraPtr pCamera);
	
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

	// This is our callback routine that will be executed on every received frame
    virtual void FrameReceived( const FramePtr pFrame);

private:
	VmbUchar_t *pImage;
    double GetTime();
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

    template <typename T>
    class ValueWithState
    {
    private:
        T m_Value;
        bool m_State;


    public:
        ValueWithState()
            : m_State( false )
        {}
        ValueWithState( T &value )
            : m_Value ( value )
            , m_State( true )
        {}
        const T& operator()() const
        {
            return m_Value;
        }
        void operator()( const T &value )
        {
            m_Value = value;
            m_State = true;
        }
        bool IsValid() const
        {
            return m_State;
        }
        void Invalidate()
        {
            m_State = false;
        }
    };
    ValueWithState<double>      m_FrameTime;
    ValueWithState<VmbUint64_t> m_FrameID;
#ifdef WIN32
    double      m_dFrequency;
#endif //WIN32
};
}}
#endif