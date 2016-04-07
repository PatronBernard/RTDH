/*=============================================================================
  Copyright (C) 2012 - 2016 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        ApiController.h

  Description: Header file for the ApiController helper class that demonstrates
               how to implement an asynchronous, continuous image acquisition
               with VimbaCPP.

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

#ifndef AVT_VMBAPI_EXAMPLES_APICONTROLLER
#define AVT_VMBAPI_EXAMPLES_APICONTROLLER

#include <string>
#include <sstream>
#include <iostream>

#ifndef UNICODE
    typedef std::string             string_type;
    typedef std::ostringstream      string_stream_type;
#else
    typedef std::wstring            string_type;
    typedef std::wostringstream     string_stream_type;
#endif

#include <VimbaCPP/Include/VimbaCPP.h>

#include "FrameObserver.h"

namespace AVT {
namespace VmbAPI {
namespace Examples {

class ApiController
{
  public:
    ApiController();
    ~ApiController();

    //
    // Starts the Vimba API and loads all transport layers
    //
    // Returns:
    //  An API status code
    //
    VmbErrorType        StartUp();

    //
    // Shuts down the API
    //
    void                ShutDown();

    //
    // Opens the given camera
    // Sets the maximum possible Ethernet packet size
    // Adjusts the image format
    // Sets up the observer that will be notified on every incoming frame
    // Calls the API convenience function to start image acquisition
    // Closes the camera in case of failure
    //
    // Parameters:
    //  [in]    rStrCameraID    The ID of the camera to open as reported by Vimba
    //
    // Returns:
    //  An API status code
    //
    VmbErrorType        StartContinuousImageAcquisition( const std::string &rStrCameraID );

    //
    // Calls the API convenience function to stop image acquisition
    // Closes the camera
    //
    // Returns:
    //  An API status code
    //
    VmbErrorType        StopContinuousImageAcquisition();

    //
    // Gets the width of a frame
    //
    // Returns:
    //  The width as integer
    //
    int                 GetWidth();

    //
    // Gets the height of a frame
    //
    // Returns:
    //  The height as integer
    //
    int                 GetHeight();

    //
    // Gets the pixel format of a frame
    //
    // Returns:
    //  The pixel format as enum
    //
    VmbPixelFormatType  GetPixelFormat();

    //
    // Gets all cameras known to Vimba
    //
    // Returns:
    //  A vector of camera shared pointers
    //
    CameraPtrVector     GetCameraList();

    //
    // Gets the oldest frame that has not been picked up yet
    //
    // Returns:
    //  A frame shared pointer
    //
    FramePtr            GetFrame();

    //
    // Queues a given frame to be filled by the API
    //
    // Parameters:
    //  [in]    pFrame          The frame to queue
    //
    // Returns:
    //  An API status code
    //    
    VmbErrorType        QueueFrame( FramePtr pFrame );

    //
    // Clears all remaining frames that have not been picked up
    //
    void                ClearFrameQueue();

    //
    // Translates Vimba error codes to readable error messages
    //
    // Parameters:
    //  [in]    eErr        The error code to be converted to string
    //
    // Returns:
    //  A descriptive string representation of the error code
    //
    string_type         ErrorCodeToMessage( VmbErrorType eErr ) const;

    //
    // Gets the version of the Vimba API
    //
    // Returns:
    //  The version as string
    //
    string_type         GetVersion() const;
  private:
    // A reference to our Vimba singleton
    VimbaSystem &m_system;
    // The currently streaming camera
    CameraPtr m_pCamera;
    // Every camera has its own frame observer
    IFrameObserverPtr m_pFrameObserver;
    // The current pixel format
    VmbInt64_t m_nPixelFormat;
    // The current width
    VmbInt64_t m_nWidth;
    // The current height
    VmbInt64_t m_nHeight;
};

}}} // namespace AVT::VmbAPI::Examples

#endif
