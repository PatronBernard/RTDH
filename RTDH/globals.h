#ifndef CAMERAMODE_H
#define CAMERAMODE_H

#define PI	3.1415926535897932384626433832795028841971693993751058209749
#define PI2 1.570796326794896619231321691639751442098584699687552910487
#include "cufftXt.h"
typedef float2 Complex;

enum cameraMode{
	cameraModeVideo,
	cameraModeReconstruct,
	cameraModeReconstructI,
	cameraModeFFT,
	cameraModeViewStoredFrame,
	cameraModeCombined
};

enum displayMode{
	displayModeMagnitude, 
	displayModePhase
};

extern cameraMode cMode;
extern displayMode dMode;
extern bool storeCurrentFrame;
extern bool addRecordedFrameToCurrent;
#endif