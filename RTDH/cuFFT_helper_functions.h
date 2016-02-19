//Written by Jan Morez (University of Antwerp,2014-2015) for a master's thesis 
//on Realtime Digital Holographic Interferometry

#ifndef CUFFT_HELPER_FUNCTIONS_H
#define CUFFT_HELPER_FUNCTIONS_H

#include "cufftXt.h"

char* cufftStrError(cufftResult_t result){
	switch (result)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS TEST";
	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	case CUFFT_INCOMPLETE_PARAMETER_LIST:
		return "CUFFT_INCOMPLETE_PARAMETER_LIST";
	case CUFFT_INVALID_DEVICE:
		return "CUFFT_INVALID_DEVICE";
	case CUFFT_PARSE_ERROR:
		return "CUFFT_PARSE_ERROR";
	case CUFFT_NO_WORKSPACE:
		return "CUFFT_NO_WORKSPACE";
	case CUFFT_NOT_IMPLEMENTED:
		return "CUFFT_NOT_IMPLEMENTED";
	case CUFFT_LICENSE_ERROR:
		return "CUFFT_LICENSE_ERROR";
	default:
		return "NO_MATCHING_CASE";
	}
}

#endif