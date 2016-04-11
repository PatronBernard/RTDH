#ifndef CAMERAMODE_H
#define CAMERAMODE_H
enum cameraMode{
	cameraModeVideo,
	cameraModeReconstruct,
	cameraModeFFT,
	cameraModeViewStoredFrame
};

enum displayMode{
	displayModeMagnitude, 
	displayModePhase
};

extern cameraMode cMode;
extern displayMode dMode;
extern bool storeCurrentFrame;
#endif