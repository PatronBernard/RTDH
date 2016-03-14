/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        LoadSaveSettings.cpp

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

#include <string>
#include <sstream>
#include <tinyxml.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "LoadSaveSettings.h"

#include "VimbaCPP/Include/VimbaCPP.h"

#ifdef max
#   undef max
#endif
namespace AVT {
namespace VmbAPI {
namespace Examples {

//Helper function to add a feature to the xml
//The xml entry will look like that:
//<rType Name="MyFeatureName">rValue</rType>
VmbErrorType AddNode(FeaturePtr pFeature, const std::string &rType, const std::string &rValue, TiXmlNode &rXmlNode)
{
    if(NULL ==pFeature)
    {
        return VmbErrorBadParameter;
    }

    VmbErrorType    err = VmbErrorSuccess;

    std::string     name;
    if(VmbErrorSuccess == err)
    {
        err = pFeature->GetName(name);
    }

    TiXmlElement featureNode(rType.c_str());

    if(VmbErrorSuccess == err)
    {
        featureNode.SetAttribute("Name", name.c_str());
        TiXmlText content(rValue.c_str());
        if(featureNode.InsertEndChild(content) == NULL)
        {
            err = VmbErrorInternalFault;
        }
    }

    if(VmbErrorSuccess == err)
    {
        if(rXmlNode.InsertEndChild(featureNode) == NULL)
        {
            err = VmbErrorInternalFault;
        }
    }

    return err;
}

//Creates a copy of a string without its leading and trailing spaces.
std::string Trim(const std::string &rString)
{
    std::string     strResult;

    size_t          first       = rString.size();
    size_t          last        = 0;

    for(size_t i = 0; i < rString.size(); i++)
    {
        if(rString[i] != ' ')
        {
            if(first >= rString.size())
            {
                first = i;
            }

            last = i;
        }
    }

    if(last >= first)
    {
        strResult = rString.substr(first, last - first + 1);
    }

    return strResult;
}

//Creates an uppercase copy of a string
std::string ToUpper(const std::string &rString)
{
    std::string strResult;

    for(size_t i = 0; i < rString.size(); i++)
    {
        strResult += toupper(rString[i]);
    }

    return strResult;
}

//Parse a string to a VmbInt64_t
VmbErrorType Parse(const std::string &rString, VmbInt64_t &rValue)
{
    //Ignore leading an trailing spaces
    std::string strTrimmed = Trim(rString);

    if(strTrimmed.size() <= 0)
    {
        return VmbErrorOther;
    }

    //Make sure that the string only contains numbers
    for(size_t i = 0; i < strTrimmed.size(); i++)
    {
        if(     (strTrimmed[i] < '0')
            ||  (strTrimmed[i] > '9'))
        {
            return VmbErrorOther;
        }
    }

    VmbInt64_t          value(0);
    std::stringstream   ss(strTrimmed);
    ss >> value;

    if(ss.fail() == true)
    {
        return VmbErrorOther;
    }

    rValue = value;

    return VmbErrorSuccess;
}

//Parse a string to a double
VmbErrorType Parse(const std::string &rString, double &rValue)
{
    //Ignore leading an trailing spaces
    std::string strTrimmed = Trim(rString);

    if(strTrimmed.size() <= 0)
    {
        return VmbErrorOther;
    }

    double              value( 0.0 );
    std::stringstream   ss(strTrimmed);
    ss >> value;

    if(ss.fail() == true)
    {
        return VmbErrorOther;
    }

    rValue = value;

    return VmbErrorSuccess;
}

//Parse a string to a bool
//"true" (case insensitive) and "1" -> true
//"false" (case insensitive) and "0" -> false
VmbErrorType Parse(const std::string &rString, bool &rValue)
{
    //Ignore leading an trailing spaces and case
    std::string strModified = ToUpper(Trim(rString));

    if(     (strModified.compare("TRUE") == 0)
        ||  (strModified.compare("1") == 0))
    {
        rValue = true;
        return VmbErrorSuccess;
    }
    else if(    (strModified.compare("FALSE") == 0)
            ||  (strModified.compare("0") == 0))
    {
        rValue = false;
        return VmbErrorSuccess;
    }

    return VmbErrorOther;
}

// class FeatureValue /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LoadSaveSettings::FeatureValue::FeatureValue(FeaturePtr pFeature)
    :   m_pFeature(pFeature)
{
}

LoadSaveSettings::FeatureValue::~FeatureValue()
{
}

FeaturePtr LoadSaveSettings::FeatureValue::GetFeature()
{
    return m_pFeature;
}

VmbErrorType LoadSaveSettings::FeatureValue::IsWritable(bool &rWritable)
{
    //First check the feature
    if(NULL == m_pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is writable
    bool            writable    = true;
    VmbErrorType    err         = m_pFeature->IsWritable(writable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    rWritable = writable;

    return VmbErrorSuccess;
}

// class IntegerValue /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LoadSaveSettings::IntegerValue::IntegerValue(FeaturePtr pFeature, VmbInt64_t targetValue)
    :   FeatureValue(pFeature)
    ,   m_TargetValue(targetValue)
{
}

VmbErrorType LoadSaveSettings::IntegerValue::IsWritable(bool &rWritable)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is writable at all
    bool            writable    = true;
    VmbErrorType    err         = pFeature->IsWritable(writable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    //Then check if the target value is within the current range and
    //matches the increment.
    if(true == writable)
    {
        VmbInt64_t minValue = 0;
        VmbInt64_t maxValue = 0;

        err = pFeature->GetRange(minValue, maxValue);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        if(     (m_TargetValue < minValue)
            ||  (m_TargetValue > maxValue))
        {
            writable = false;
        }
        else
        {
            VmbInt64_t incValue = 0;
            err = pFeature->GetIncrement(incValue);
            if(VmbErrorSuccess != err)
            {
                return err;
            }

            if(incValue < 1)
            {
                return VmbErrorInternalFault;
            }

            if(((m_TargetValue - minValue) % incValue) != 0)
            {
                writable = false;
            }
        }
    }

    rWritable = writable;

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::IntegerValue::IsTargetValue(bool &rTargetValue)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is readable at all
    bool            readable    = false;
    VmbErrorType    err         = pFeature->IsReadable(readable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }
    
    //Now read the current value and compare it to our target value
    if(true == readable)
    {
        VmbInt64_t value = 0;
        err = pFeature->GetValue(value);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        if(value == m_TargetValue)
        {
            rTargetValue = true;
        }
        else
        {
            rTargetValue = false;
        }
    }
    else
    {
        rTargetValue = false;
    }

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::IntegerValue::ApplyTargetValue()
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    VmbErrorType err = pFeature->SetValue(m_TargetValue);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    return VmbErrorSuccess;
}

// class FloatValue /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LoadSaveSettings::FloatValue::FloatValue(FeaturePtr pFeature, double targetValue)
    :   FeatureValue(pFeature)
    ,   m_TargetValue(targetValue)
{
}

VmbErrorType LoadSaveSettings::FloatValue::IsWritable(bool &rWritable)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is writable at all
    bool writable = true;
    VmbErrorType err = pFeature->IsWritable(writable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    //Then check if the target value is within the current range.
    if(true == writable)
    {
        double minValue = 0.0;
        double maxValue = 0.0;
    
        err = pFeature->GetRange(minValue, maxValue);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        if(     (m_TargetValue < minValue)
            ||  (m_TargetValue > maxValue))
        {
            writable = false;
        }
    }

    rWritable = writable;

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::FloatValue::IsTargetValue(bool &rTargetValue)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is readable at all
    bool readable = false;
    VmbErrorType err = pFeature->IsReadable(readable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }
    
    //Now read the current value and compare it to our target value
    if(true == readable)
    {
        double value = 0;
        err = pFeature->GetValue(value);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        //Let's assume we don't want an exact match but
        //at least a very close match.
        if(abs(value - m_TargetValue) < 1e-8) //Match with absolute precision
        {
            rTargetValue = true;
        }
        else if((abs(value - m_TargetValue) / std::max(abs(value), abs(m_TargetValue))) < 1e-8) //Match with relative precision
        {
            rTargetValue = true;
        }
        else //No match
        {
            rTargetValue = false;
        }
    }
    else
    {
        rTargetValue = false;
    }

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::FloatValue::ApplyTargetValue()
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Write value to feature
    VmbErrorType err = pFeature->SetValue(m_TargetValue);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    return VmbErrorSuccess;
}

// class EnumerationValue /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LoadSaveSettings::EnumerationValue::EnumerationValue(FeaturePtr pFeature, std::string targetValue)
    :   FeatureValue(pFeature)
    ,   m_TargetValue(targetValue)
{
}

VmbErrorType LoadSaveSettings::EnumerationValue::IsWritable(bool &rWritable)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is writable at all
    bool            writable    = true;
    VmbErrorType    err         = pFeature->IsWritable(writable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    //Check if the target value is one of our enum entries
    if(true == writable)
    {
        StringVector enumValues;
        err = pFeature->GetValues(enumValues);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        bool valueFound = false;
        for(    StringVector::iterator i = enumValues.begin();
                i != enumValues.end();
                ++i)
        {
            if(m_TargetValue.compare(*i) == 0)
            {
                //Check if this enum entry is available at all
                bool available = false;
                err = pFeature->IsValueAvailable(i->c_str(), available);
                if(VmbErrorSuccess != err)
                {
                    return err;
                }

                if(true == available)
                {
                    valueFound = true;
                    break;
                }
            }
        }

        if(false == valueFound)
        {
            writable = false;
        }
    }

    rWritable = writable;

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::EnumerationValue::IsTargetValue(bool &rTargetValue)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is readable at all
    bool            readable    = false;
    VmbErrorType    err         = pFeature->IsReadable(readable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }
    
    //Now read the current value and compare it to our target value
    if(true == readable)
    {
        std::string value;
        err = pFeature->GetValue(value);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        if(value.compare(m_TargetValue) == 0)
        {
            rTargetValue = true;
        }
        else
        {
            rTargetValue = false;
        }
    }
    else
    {
        rTargetValue = false;
    }

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::EnumerationValue::ApplyTargetValue()
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Write value to feature
    VmbErrorType err = pFeature->SetValue(m_TargetValue.c_str());
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    return VmbErrorSuccess;
}

// class StringValue /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LoadSaveSettings::StringValue::StringValue(FeaturePtr pFeature, std::string targetValue)
    :   FeatureValue(pFeature)
    ,   m_TargetValue(targetValue)
{
}

VmbErrorType LoadSaveSettings::StringValue::IsTargetValue(bool &rTargetValue)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is readable at all
    bool            readable    = false;
    VmbErrorType    err         = pFeature->IsReadable(readable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }
    
    //Now read the current value and compare it to our target value
    if(true == readable)
    {
        std::string value;
        err = pFeature->GetValue(value);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        if(value.compare(m_TargetValue) == 0)
        {
            rTargetValue = true;
        }
        else
        {
            rTargetValue = false;
        }
    }
    else
    {
        rTargetValue = false;
    }

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::StringValue::ApplyTargetValue()
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Write value to feature
    VmbErrorType err = pFeature->SetValue(m_TargetValue.c_str());
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    return VmbErrorSuccess;
}

// class BooleanValue /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LoadSaveSettings::BooleanValue::BooleanValue(FeaturePtr pFeature, bool targetValue)
    :   FeatureValue(pFeature)
    ,   m_TargetValue(targetValue)
{
}

VmbErrorType LoadSaveSettings::BooleanValue::IsTargetValue(bool &rTargetValue)
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Check if it is readable at all
    bool readable = false;
    VmbErrorType err = pFeature->IsReadable(readable);
    if(VmbErrorSuccess != err)
    {
        return err;
    }
    
    //Now read the current value and compare it to our target value
    if(true == readable)
    {
        bool value = false;
        err = pFeature->GetValue(value);
        if(VmbErrorSuccess != err)
        {
            return err;
        }

        if(value == m_TargetValue)
        {
            rTargetValue = true;
        }
        else
        {
            rTargetValue = false;
        }
    }
    else
    {
        rTargetValue = false;
    }

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::BooleanValue::ApplyTargetValue()
{
    //First try to get the feature
    FeaturePtr pFeature = GetFeature();
    if(NULL == pFeature)
    {
        return VmbErrorBadHandle;
    }

    //Write value to feature
    VmbErrorType err = pFeature->SetValue(m_TargetValue);
    if(VmbErrorSuccess != err)
    {
        return err;
    }

    return VmbErrorSuccess;
}

VmbErrorType LoadSaveSettings::SaveToFile(CameraPtr pCamera, const char *pFileName, bool ignoreStreamable)
{
    //Check parameters
    if(     (NULL == pCamera)
        ||  (NULL == pFileName))
    {
        return VmbErrorBadParameter;
    }

    VmbErrorType err = VmbErrorSuccess;

    //All settings are embedded into a settings node
    TiXmlElement settingsNode("Settings");

    //Get camera id
    std::string id;
    if(VmbErrorSuccess == err)
    {
        err = pCamera->GetID(id);
    }

    //Set camera ID as attribut in settings node
    if(VmbErrorSuccess == err)
    {
        settingsNode.SetAttribute("ID", id.c_str());
    }

    //Get camera name
    std::string name;
    if(VmbErrorSuccess == err)
    {
        err = pCamera->GetName(name);
    }

    //Set camera name as attribute in settings node
    if(VmbErrorSuccess == err)
    {
        settingsNode.SetAttribute("Name", name.c_str());
    }

    //Get camera model
    std::string model;
    if(VmbErrorSuccess == err)
    {
        err = pCamera->GetModel(model);
    }

    //Set camera model as attribute in settings node
    if(VmbErrorSuccess == err)
    {
        settingsNode.SetAttribute("Model", model.c_str());
    }

    FeaturePtrVector features;

    //Get list of features
    if(VmbErrorSuccess == err)
    {
        err = pCamera->GetFeatures(features);
    }

    if(VmbErrorSuccess == err)
    {
        //Iterate over all features and add them to the xml
        for(    FeaturePtrVector::iterator i = features.begin();
                (i != features.end()) && (VmbErrorSuccess == err);
                ++i)
        {
            FeaturePtr pFeature = *i;

            if(VmbErrorSuccess == err)
            {
                if(NULL == pFeature)
                {
                    err = VmbErrorBadHandle;
                }
            }

            //Check if the current feature is readable now
            bool readable = false;
            if(VmbErrorSuccess == err)
            {
                err = pFeature->IsReadable(readable);
            }

            if(VmbErrorSuccess == err)
            {
                if(false == readable)
                {
                    continue;
                }
            }

            if(false == ignoreStreamable)
            {
                bool isStreamable = false;
                if(VmbErrorSuccess == err)
                {
                    err = pFeature->IsStreamable(isStreamable);
                }

                if(VmbErrorSuccess == err)
                {
                    if(false == isStreamable)
                    {
                        continue;
                    }
                }
            }

            //Now get the current features data type
            VmbFeatureDataType type = VmbFeatureDataUnknown;
            if(VmbErrorSuccess == err)
            {
                err = pFeature->GetDataType(type);
            }

            if(VmbErrorSuccess == err)
            {
                //Only write features with the following data types:
                //Integer, Float, Enumeration, String and Boolean
                switch(type)
                {
                //Report unsupported feature data types
                default:
                case VmbFeatureDataUnknown:
                case VmbFeatureDataNone:
                    err = VmbErrorWrongType;
                    break;

                //Ignore some feature data types
                case VmbFeatureDataCommand:
                case VmbFeatureDataRaw:
                    break;

                //Add xml nodes for supported feature data types
                case VmbFeatureDataInt:
                    {
                        VmbInt64_t value = 0;
                        err = pFeature->GetValue(value);
                        if(VmbErrorSuccess == err)
                        {
                            std::stringstream ss;
                            ss << value;
                            std::string strValue = ss.str();

                            err = AddNode(pFeature, "Integer", strValue, settingsNode);
                        }
                    }
                    break;

                case VmbFeatureDataFloat:
                    {
                        double value = 0.0;
                        err = pFeature->GetValue(value);
                        if(VmbErrorSuccess == err)
                        {
                            std::stringstream ss;
                            ss.precision(15);
                            ss << value;
                            std::string strValue = ss.str();

                            err = AddNode(pFeature, "Float", strValue, settingsNode);
                        }
                    }
                    break;

                case VmbFeatureDataEnum:
                    {
                        std::string value;
                        err = pFeature->GetValue(value);
                        if(VmbErrorSuccess == err)
                        {
                            err = AddNode(pFeature, "Enumeration", value, settingsNode);
                        }
                    }
                    break;

                case VmbFeatureDataString:
                    {
                        std::string value;
                        err = pFeature->GetValue(value);
                        if(VmbErrorSuccess == err)
                        {
                            err = AddNode(pFeature, "String", value, settingsNode);
                        }
                    }
                    break;

                case VmbFeatureDataBool:
                    {
                        bool value = false;
                        err = pFeature->GetValue(value);
                        if(VmbErrorSuccess == err)
                        {
                            std::string strValue;
                            if(true == value)
                            {
                                strValue = "True";
                            }
                            else
                            {
                                strValue = "False";
                            }

                            err = AddNode(pFeature, "Boolean", strValue, settingsNode);
                        }
                    }
                    break;
                }
            }
        }
    }

    TiXmlDocument xmlDocument;

    //Add a simple xml declaration such as <?xml version="1.0"?>
    if(VmbErrorSuccess == err)
    {
        TiXmlDeclaration declaration("1.0", "", "");
        if(xmlDocument.InsertEndChild(declaration) == NULL)
        {
            err = VmbErrorInternalFault;
        }
    }

    //Add the settings node to the xml document
    if(VmbErrorSuccess == err)
    {
        if(xmlDocument.InsertEndChild(settingsNode) == NULL)
        {
            err = VmbErrorInternalFault;
        }
    }

    //Write the xml document to file
    if(VmbErrorSuccess == err)
    {
        if(xmlDocument.SaveFile(pFileName) == false)
        {
            err = VmbErrorOther;
        }
    }

    return err;
}

VmbErrorType LoadSaveSettings::LoadFromFile(CameraPtr pCamera, const char *pFileName, StringVector &rLoadedFeatures, StringVector &rMissingFeatures, bool ignoreStreamable, VmbUint32_t maxIterations)
{
    //Check parameters
    if(     (NULL == pCamera)
        ||  (NULL == pFileName)
        ||  (maxIterations < 1))
    {
        return VmbErrorBadParameter;
    }

    VmbErrorType err = VmbErrorSuccess;

    //Don't ignore whitespaces in xml because restoring
    //string features may fail otherwise.
    TiXmlBase::SetCondenseWhiteSpace(false);

    TiXmlDocument xmlDocument;

    //Load the xml document from file
    if(VmbErrorSuccess == err)
    {
        if(xmlDocument.LoadFile(pFileName) == false)
        {
            err = VmbErrorResources;
        }
    }

    //Get the settings node
    TiXmlElement *pSettingsElement = NULL;
    if(VmbErrorSuccess == err)
    {
        pSettingsElement = xmlDocument.FirstChildElement("Settings");
        if(NULL == pSettingsElement)
        {
            err = VmbErrorOther;
        }
    }

    //Get camera model
    std::string model;
    if(VmbErrorSuccess == err)
    {
        err = pCamera->GetModel(model);
    }

    //Check if the camera model matches the one from the xml file
    if(VmbErrorSuccess == err)
    {
        const char *pModel = pSettingsElement->Attribute("Model");
        if(NULL == pModel)
        {
            err = VmbErrorOther;
        }
        else
        {
            if(ToUpper(Trim(model)).compare(ToUpper(Trim(pModel))) != 0)
            {
                err = VmbErrorNotFound;
            }
        }
    }

    StringVector loadedFeatures;
    StringVector missingFeatures;

    //First load all features from xml
    std::vector<FeatureValuePtr> featureValues;
    if(VmbErrorSuccess == err)
    {
        TiXmlNode *pFeatureNode = pSettingsElement->FirstChild();
        while(NULL != pFeatureNode)
        {
            //Try to convert the xml node to an xml element
            TiXmlElement *pFeatureElement = pFeatureNode->ToElement();
            if(NULL == pFeatureElement)
            {
                err = VmbErrorOther;
                break;
            }

            const char *pType = pFeatureElement->Value();
            if(NULL == pType)
            {
                err = VmbErrorOther;
                break;
            }

            //Get the feature name from the attribute
            const char *pName = pFeatureElement->Attribute("Name");
            if(NULL == pName)
            {
                err = VmbErrorOther;
                break;
            }

            TiXmlNode *pValueNode = pFeatureElement->FirstChild();
            if(NULL == pValueNode)
            {
                err = VmbErrorOther;
                break;
            }

            //Get the feature target value as a string
            const char *pValue = pValueNode->Value();
            if(NULL == pValue)
            {
                err = VmbErrorOther;
                break;
            }

            //Try to find the feature with the given name
            FeaturePtr pFeature;
            err = pCamera->GetFeatureByName(pName, pFeature);
            if(VmbErrorSuccess == err)
            {
                if(NULL == pFeature)
                {
                    err = VmbErrorBadHandle;
                    break;
                }

                bool loadFeature = true;
                if(false == ignoreStreamable)
                {
                    bool isStreamable = false;
                    err = pFeature->IsStreamable(isStreamable);
                    if(VmbErrorSuccess != err)
                    {
                        break;
                    }

                    if(false == isStreamable)
                    {
                        loadFeature = false;
                    }
                }

                if(true == loadFeature)
                {
                    //Create a feature value for the current feature
                    //by parsing the value depending on the features
                    //data type.
                    FeatureValue *pFeatureValue = NULL;
                    if(strcmp(pType, "Integer") == 0)
                    {
                        VmbInt64_t value = 0;
                        err = Parse(pValue, value);
                        if(VmbErrorSuccess != err)
                        {
                            break;
                        }

                        pFeatureValue = new IntegerValue(pFeature, value);
                    }
                    else if(strcmp(pType, "Float") == 0)
                    {
                        double value = 0.0;
                        err = Parse(pValue, value);
                        if(VmbErrorSuccess != err)
                        {
                            break;
                        }

                        pFeatureValue = new FloatValue(pFeature, value);
                    }
                    else if(strcmp(pType, "Enumeration") == 0)
                    {
                        pFeatureValue = new EnumerationValue(pFeature, Trim(pValue));
                    }
                    else if(strcmp(pType, "String") == 0)
                    {
                        pFeatureValue = new StringValue(pFeature, pValue);
                    }
                    else if(strcmp(pType, "Boolean") == 0)
                    {
                        bool value = false;
                        err = Parse(pValue, value);
                        if(VmbErrorSuccess != err)
                        {
                            break;
                        }
                
                        pFeatureValue = new BooleanValue(pFeature, value);
                    }
                    else
                    {
                        //Report an error if we found an unsupported
                        //feature data type.
                        err = VmbErrorOther;
                        break;
                    }

                    //Check if we were able to allocate the feature value
                    if(NULL == pFeatureValue)
                    {
                        err = VmbErrorResources;
                        break;
                    }

                    //Assign the feature value to a shared pointer
                    FeatureValuePtr pFeatureValueShared(pFeatureValue);

                    //Add the new feature value to the list of feature values
                    featureValues.push_back(pFeatureValueShared);
                }
                else
                {
                    //We directly add the feature to the missing features list
                    //if feature is not streamable.
                    missingFeatures.push_back(pName);
                }
            }
            else
            {
                //We directly add the feature to the missing features list
                //if no feature exists with the given name.
                missingFeatures.push_back(pName);
            }

            //Continue iterating over the xml entries
            pFeatureNode = pFeatureElement->NextSibling();
        }
    }

    //Now we try to write all features into the camera
    if(VmbErrorSuccess == err)
    {
        VmbUint32_t iteration           = 0;        //Counter for retries
        bool        featuresComplete    = false;    //Is true if all features have been set
        bool        featuresWritten     = true;     //Is true if any feature has been written during the last iteration

        while(      (VmbErrorSuccess == err)        //Only continue if there was no error
                &&  (false == featuresComplete)     //Only iterate if we are not done yet
                &&  (true == featuresWritten)       //Only iterate if we are not stuck (no features left that can be changed)
                &&  (iteration < maxIterations))    //Only iterate until we reach the maximum number of interations/retries
        {
            featuresComplete    = true;
            featuresWritten     = false;

            //Iterate over all feature values and try to set them
            for(    std::vector<FeatureValuePtr>::const_iterator i = featureValues.begin();
                    (i != featureValues.end()) && (VmbErrorSuccess == err);
                    ++i)
            {
                FeatureValuePtr pFeatureValue = *i;

                //We only set a feature if it doesn't already contain the target value
                bool targetValue = false;
                err = pFeatureValue->IsTargetValue(targetValue);
                if(VmbErrorSuccess != err)
                {
                    break;
                }

                if(false == targetValue)
                {
                    //Remember that there is at least one feature to be done
                    featuresComplete = false;

                    //We only set a feature if the target value can be written at the moment
                    bool writable = false;
                    err = pFeatureValue->IsWritable(writable);
                    if(VmbErrorSuccess != err)
                    {
                        break;
                    }

                    if(true == writable)
                    {
                        //Write the target value to the feature
                        err = pFeatureValue->ApplyTargetValue();
                        if(VmbErrorSuccess != err)
                        {
                            break;
                        }

                        //Remember that we changed at least one feature
                        featuresWritten = true;
                    }
                }
            }

            iteration++;
        }
    }

    //Finally check the contents of all features once more to make sure that all features
    //now contain the according target value.
    if(VmbErrorSuccess == err)
    {
        for(    std::vector<FeatureValuePtr>::const_iterator i = featureValues.begin();
                i != featureValues.end();
                ++i)
        {
            FeatureValuePtr pFeatureValue   = *i;

            FeaturePtr      pFeature        = pFeatureValue->GetFeature();

            //Get the feature name to be able to add the feature to
            //either the loadedFeatures or the missingFeatures.
            std::string name;
            err = pFeature->GetName(name);
            if(VmbErrorSuccess != err)
            {
                break;
            }

            //Check if the current feature contains the target value
            bool targetValue = false;
            err = pFeatureValue->IsTargetValue(targetValue);
            if(VmbErrorSuccess != err)
            {
                break;
            }

            //Add the feature to one of our lists depending on if it contains the target value.
            if(true == targetValue)
            {
                loadedFeatures.push_back(name);
            }
            else
            {
                missingFeatures.push_back(name);
            }
        }
    }

    //Return the feature lists to the user if there
    //was no error.
    if(VmbErrorSuccess == err)
    {
        rLoadedFeatures     = loadedFeatures;
        rMissingFeatures    = missingFeatures;
    }

    return err;
}

}}} // namespace AVT::VmbAPI::Examples

