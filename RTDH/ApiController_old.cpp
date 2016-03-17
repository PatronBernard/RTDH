/*=============================================================================
  Copyright (C) 2013 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        ApiController.cpp

  Description: Implementation file for the ApiController helper class that
               demonstrates how to implement a synchronous single image
               acquisition with VimbaCPP.

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

#include <sstream>
#include <iostream>

#include "ApiController.h"
#include "Common/StreamSystemInfo.h"
#include "Common/ErrorCodeToMessage.h"

namespace AVT {
namespace VmbAPI {
namespace Examples {

enum { NUM_FRAMES = 3, };

ApiController::ApiController()
    // Get a reference to the Vimba singleton
    : m_system ( VimbaSystem::GetInstance() )
{
}

ApiController::~ApiController()
{
}


VmbErrorType ApiController::StartUp()
{
    return m_system.Startup();
}

void ApiController::ShutDown()
{
    // Release Vimba
    m_system.Shutdown();
}

VmbErrorType ApiController::AcquireSingleImage( const std::string &rStrCameraID, FramePtr &rpFrame )
{
    // Open the desired camera by its ID
    VmbErrorType res = m_system.OpenCameraByID( rStrCameraID.c_str(), VmbAccessModeFull, m_pCamera );
    if ( VmbErrorSuccess == res )
    {
        // Set the GeV packet size to the highest possible value
        // (In this example we do not test whether this cam actually is a GigE cam)
        FeaturePtr pCommandFeature;
        if ( VmbErrorSuccess == m_pCamera->GetFeatureByName( "GVSPAdjustPacketSize", pCommandFeature ))
        {
            if ( VmbErrorSuccess == pCommandFeature->RunCommand() )
            {
                bool bIsCommandDone = false;
                do
                {
                    if ( VmbErrorSuccess != pCommandFeature->IsCommandDone( bIsCommandDone ))
                    {
                        break;
                    }
                } while ( false == bIsCommandDone );
            }
        }
        FeaturePtr pFormatFeature;
        // Set pixel format. For the sake of simplicity we only support Mono and BGR in this example.
        res = m_pCamera->GetFeatureByName( "PixelFormat", pFormatFeature );
        if ( VmbErrorSuccess == res )
        {
            // Try to set BGR
            res = pFormatFeature->SetValue( VmbPixelFormatRgb8 );
            if ( VmbErrorSuccess != res )
            {
                // Fall back to Mono
                res = pFormatFeature->SetValue( VmbPixelFormatMono8 );
            }

            if ( VmbErrorSuccess == res )
            {
                // Acquire
                res = m_pCamera->AcquireSingleImage( rpFrame, 5000 );
            }
        }

        m_pCamera->Close();
    }

    return res;
}

CameraPtrVector ApiController::GetCameraList()
{
    CameraPtrVector cameras;
    // Get all known cameras
    if ( VmbErrorSuccess == m_system.GetCameras( cameras ))
    {
        // And return them
        return cameras;
    }
    return CameraPtrVector();
}
// Translates Vimba error codes to readable error messages
std::string ApiController::ErrorCodeToMessage( VmbErrorType eErr ) const
{
    return AVT::VmbAPI::Examples::ErrorCodeToMessage( eErr );
}
// Translates Vimba error codes to readable error messages

std::string ApiController::GetVersion() const
{
    std::ostringstream os;
    os<<m_system;
    return os.str();
}


}}} // namespace AVT::VmbAPI::Examples
