/*=============================================================================
  Copyright (C) 2013 Allied Vision Technologies.  All Rights Reserved.

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
#include "VimbaCPP/Include/VimbaCPP.h"
#include "ProgramConfig.h"


#ifdef WIN32
#include <Windows.h>
#endif //WIN32


namespace AVT {
namespace VmbAPI {
namespace Examples {


class FrameObserver : virtual public IFrameObserver
{
public:
    // We pass the camera that will deliver the frames to the constructor
    FrameObserver( CameraPtr pCamera, FrameInfos eFrameInfos, ColorProcessing eColorProcessing );
    
    // This is our callback routine that will be executed on every received frame
    virtual void FrameReceived( const FramePtr pFrame);


private:
	VmbUchar_t *pImage;
    void ShowFrameInfos( const FramePtr & );
    double GetTime();
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
    const FrameInfos            m_eFrameInfos;
    const ColorProcessing       m_eColorProcessing;
    ValueWithState<double>      m_FrameTime;
    ValueWithState<VmbUint64_t> m_FrameID;
#ifdef WIN32
    double      m_dFrequency;
#endif //WIN32
};

}}} // namespace AVT::VmbAPI::Examples

#endif
