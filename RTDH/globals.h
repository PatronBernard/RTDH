#ifndef CAMERAMODE_H
#define CAMERAMODE_H
enum cameraMode{
	cameraModeVideo,
	cameraModeReconstruct,
	cameraModeFFT
};

extern cameraMode cMode;
extern bool storeCurrentFrame;
#endif