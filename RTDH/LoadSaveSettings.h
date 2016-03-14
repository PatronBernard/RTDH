/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        LoadSaveSettings.h

  Description: The LoadSaveSettings example will demonstrate how to save the
               features from a camera to a fila and load them back from file
               using VimbaCPP.
               

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

#ifndef AVT_VIMBA_EXAMPLES_LOADSAVESETTINGS
#define AVT_VIMBA_EXAMPLES_LOADSAVESETTINGS

#include "VimbaCPP/Include/VimbaCPP.h"

namespace AVT {
namespace VmbAPI {
namespace Examples {

class LoadSaveSettings
{
private:
    //A class to manage the target value
    //of a feature.
    class FeatureValue
    {
    private:
        FeaturePtr  m_pFeature;

    public:
        FeatureValue(FeaturePtr pFeature);
        virtual ~FeatureValue();

        //Returns the according feature
        FeaturePtr GetFeature();

        //Returns true if feature is writable at the moment
        virtual VmbErrorType IsWritable(bool &rWritable);
        //Returns true if feature currently contains desired target value
        virtual VmbErrorType IsTargetValue(bool &rTargetValue) = 0;
        //Writes target value into the feature
        virtual VmbErrorType ApplyTargetValue() = 0;
    };

    //A class to manage the target value
    //of an integer feature.
    class IntegerValue : public FeatureValue
    {
    private:
        VmbInt64_t m_TargetValue;

    public:
        IntegerValue(FeaturePtr pFeature, VmbInt64_t targetValue);

        //Returns true if feature is writable at the moment
        virtual VmbErrorType IsWritable(bool &rWritable);
        //Returns true if feature currently contains desired target value
        virtual VmbErrorType IsTargetValue(bool &rTargetValue);
        //Writes target value into the feature
        virtual VmbErrorType ApplyTargetValue();
    };

    //A class to manage the target value
    //of an float feature.
    class FloatValue : public FeatureValue
    {
    private:
        double m_TargetValue;

    public:
        FloatValue(FeaturePtr pFeature, double targetValue);

        //Returns true if feature is writable at the moment
        virtual VmbErrorType IsWritable(bool &rWritable);
        //Returns true if feature currently contains desired target value
        virtual VmbErrorType IsTargetValue(bool &rTargetValue);
        //Writes target value into the feature
        virtual VmbErrorType ApplyTargetValue();
    };

    //A class to manage the target value
    //of an enumeration feature.
    class EnumerationValue : public FeatureValue
    {
    private:
        std::string m_TargetValue;

    public:
        EnumerationValue(FeaturePtr pFeature, std::string targetValue);

        //Returns true if feature is writable at the moment
        virtual VmbErrorType IsWritable(bool &rWritable);
        //Returns true if feature currently contains desired target value
        virtual VmbErrorType IsTargetValue(bool &rTargetValue);
        //Writes target value into the feature
        virtual VmbErrorType ApplyTargetValue();
    };

    //A class to manage the target value
    //of an string feature.
    class StringValue : public FeatureValue
    {
    private:
        std::string m_TargetValue;

    public:
        StringValue(FeaturePtr pFeature, std::string targetValue);

        //Returns true if feature currently contains desired target value
        virtual VmbErrorType IsTargetValue(bool &rTargetValue);
        //Writes target value into the feature
        virtual VmbErrorType ApplyTargetValue();
    };

    //A class to manage the target value
    //of an boolean feature.
    class BooleanValue : public FeatureValue
    {
    private:
        bool m_TargetValue;

    public:
        BooleanValue(FeaturePtr pFeature, bool targetValue);

        //Returns true if feature currently contains desired target value
        virtual VmbErrorType IsTargetValue(bool &rTargetValue);
        //Writes target value into the feature
        virtual VmbErrorType ApplyTargetValue();
    };

    typedef SP_DECL( FeatureValue ) FeatureValuePtr;

public:
    //Save current camera settings to xml file
    static VmbErrorType SaveToFile(CameraPtr pCamera, const char *pFileName, bool ignoreStreamable = false);

    //Load settings from xml file and then set them in the camera
    //Parameters:
    //rLoadedFeatures:  Will contain the features that have been applied to the camera successfully (valid only if function returns VmbErrorSuccess)
    //rMissingFeatures: Contains the features that couldn't be set (valid only if function returns VmbErrorSuccess)
    //maxIterations:    Maximum number of interactions (retries) to set all features
    static VmbErrorType LoadFromFile(CameraPtr pCamera, const char *pFileName, StringVector &rLoadedFeatures, StringVector &rMissingFeatures, bool ignoreStreamable = false, VmbUint32_t maxIterations = 5);
};

}}} // namespace AVT::VmbAPI::Examples

#endif //AVT_VIMBA_EXAMPLES_LOADSAVESETTINGS
