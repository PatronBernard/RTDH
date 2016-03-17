/*=============================================================================
  Copyright (C) 2013 Allied Vision Technologies.  All Rights Reserved.

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

#include <iostream>
#include <iomanip>
#ifdef WIN32
#include <Windows.h>
#else
#include <time.h>
#endif //WIN32

#include "FrameObserver.h"

#include "TransformImage.h"

namespace AVT {
namespace VmbAPI {
namespace Examples {

FrameObserver::FrameObserver( CameraPtr pCamera, FrameInfos eFrameInfos, ColorProcessing eColorProcessing )
    :   IFrameObserver( pCamera )
    ,   m_eFrameInfos( eFrameInfos )
    , m_eColorProcessing( eColorProcessing )
#ifdef WIN32
    ,   m_dFrequency( 0.0 )
#endif //WIN32
{
#ifdef WIN32
    LARGE_INTEGER nFrequency;
    QueryPerformanceFrequency( &nFrequency );
    m_dFrequency = (double)nFrequency.QuadPart;
#endif //WIN32
}

double FrameObserver::GetTime()
{
    double dTime = 0.0;

#ifdef WIN32
    LARGE_INTEGER nCounter;
    QueryPerformanceCounter( &nCounter );
    dTime = ( (double)nCounter.QuadPart ) / m_dFrequency;
#else
    //clock_t nTime = times(NULL);
    //dTime = ((double)(nTime) * 10000.0) / ((double)CLOCKS_PER_SEC);
    struct timespec now;
    clock_gettime( CLOCK_REALTIME, &now );
    dTime = ( (double)now.tv_sec ) + ( (double)now.tv_nsec ) / 1000000000.0;
#endif //WIN32

    return dTime;
}

void PrintFrameInfo( const FramePtr &pFrame )
{
    std::cout<<" Size:";
    VmbUint32_t     nWidth = 0;
    VmbErrorType    res;
    res = pFrame->GetWidth(nWidth);
    if( VmbErrorSuccess == res )
    {
        std::cout<<nWidth;
    }
    else
    {
        std::cout<<"?";
    }

    std::cout<<"x";
    VmbUint32_t nHeight = 0;
    res = pFrame->GetHeight(nHeight);
    if( VmbErrorSuccess == res )
    {
        std::cout<< nHeight;
    }
    else
    {
        std::cout<<"?";
    }

    std::cout<<" Format:";
    VmbPixelFormatType ePixelFormat = VmbPixelFormatMono8;
    res = pFrame->GetPixelFormat( ePixelFormat );
    if( VmbErrorSuccess == res )
    {
        std::cout<<"0x"<<std::hex<<ePixelFormat<<std::dec;
    }
    else
    {
        std::cout<<"?";
    }
}
void PrintFrameStatus( VmbFrameStatusType eFrameStatus )
{
    switch( eFrameStatus )
    {
    case VmbFrameStatusComplete:
        std::cout<<"Complete";
        break;

    case VmbFrameStatusIncomplete:
        std::cout<<"Incomplete";
        break;

    case VmbFrameStatusTooSmall:
        std::cout<<"Too small";
        break;

    case VmbFrameStatusInvalid:
        std::cout<<"Invalid";
        break;

    default:
        std::cout<<"unknown frame status";
        break;
    }
}

void FrameObserver::ShowFrameInfos( const FramePtr &pFrame ) 
{
    bool                bShowFrameInfos     = false;
    VmbUint64_t         nFrameID            = 0;
    bool                bFrameIDValid       = false;
    VmbFrameStatusType  eFrameStatus        = VmbFrameStatusComplete;
    bool                bFrameStatusValid   = false;
    VmbErrorType        res                 = VmbErrorSuccess;
    double              dFPS                = 0.0;
    bool                bFPSValid           = false;
    VmbUint64_t         nFramesMissing      = 0;
    if( FrameInfos_Show == m_eFrameInfos )
    {
        bShowFrameInfos = true;
    }

    res = pFrame->GetFrameID( nFrameID );
    if( VmbErrorSuccess == res )
    {
        bFrameIDValid = true;

        if( m_FrameID.IsValid() )
        {
            if( nFrameID != ( m_FrameID() + 1 ) )
            {
                nFramesMissing = nFrameID - m_FrameID() - 1;
                if( 1 == nFramesMissing )
                {
                    std::cout<<"1 missing frame detected\n";
                }
                else
                {
                    std::cout<<nFramesMissing<<"missing frames detected\n";
                }
            }
        }

        m_FrameID( nFrameID );
        double dFrameTime = GetTime();
        if(     ( m_FrameTime.IsValid() )
            &&  ( 0 == nFramesMissing ) )
        {
            double dTimeDiff = dFrameTime - m_FrameTime();
            if( dTimeDiff > 0.0 )
            {
                dFPS = 1.0 / dTimeDiff;
                bFPSValid = true;
            }
            else
            {
                bShowFrameInfos = true;
            }
        }

        m_FrameTime( dFrameTime );
    }
    else
    {
        bShowFrameInfos = true;
        m_FrameID.Invalidate();
        m_FrameTime.Invalidate();
    }

    res = pFrame->GetReceiveStatus( eFrameStatus );
    if( VmbErrorSuccess == res )
    {
        bFrameStatusValid = true;

        if( VmbFrameStatusComplete != eFrameStatus )
        {
            bShowFrameInfos = true;
        }
    }
    else
    {
        bShowFrameInfos = true;
    }
    if( bShowFrameInfos )
    {
        std::cout<<"Frame ID:";
        if( bFrameIDValid )
        {
            std::cout<<nFrameID;
        }
        else
        {
            std::cout<<"?";
        }

        std::cout<<" Status:";
        if( bFrameStatusValid )
        {
            PrintFrameStatus( eFrameStatus);
        }
        else
        {
            std::cout<<"?";
        }
        PrintFrameInfo( pFrame );
        
        std::cout<<" FPS:";
        if( bFPSValid )
        {
            std::streamsize s = std::cout.precision();
            std::cout<<std::fixed<<std::setprecision(2)<<dFPS<<std::setprecision(s);
        }
        else
        {
            std::cout<<"?";
        }

        std::cout<<"\n";
    }
    else
    {
        std::cout<<".";
    }
}
//Modified so it also reconstructs shit
void FrameObserver::FrameReceived( const FramePtr pFrame)
{
    if(! SP_ISNULL( pFrame ) )
    {
        VmbFrameStatusType status;
        VmbErrorType Result;
        Result = SP_ACCESS( pFrame)->GetReceiveStatus( status);
        if( VmbErrorSuccess == Result && VmbFrameStatusComplete == status)
        {
			 std::cout<< "Received a frame! \n";
			 pFrame->GetImage(pImage);
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
}}} // namespace AVT::VmbAPI::Examples
