/****************************************************************************
			Description:	Defines the StereoProcess Class

			Classes:		StereoProcess

			Project:		MATE 2022

			Copyright 2022 MST Design Team - Underwater Robotics.
****************************************************************************/
#ifndef StereoProcess_h
#define StereoProcess_h

#include <cstdio>
#include <string>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>

#include "VideoGet.h"
#include "FPS.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <networktables/NetworkTableInstance.h>
#include <vision/VisionPipeline.h>
#include <vision/VisionRunner.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>
#include <cameraserver/CameraServer.h>

using namespace cv;
using namespace cs;
using namespace nt;
using namespace frc;
using namespace wpi;
using namespace std;
///////////////////////////////////////////////////////////////////////////////


class StereoProcess
{
public:
    // Declare class methods.
    StereoProcess();
    ~StereoProcess();
    void Process(Mat &leftFrame, Mat &rightFrame, Mat &stereoImg, bool &enableStereoVision, VideoGet &VideoGetter, shared_timed_mutex &MutexGet, shared_timed_mutex &MutexShow);
    void SetIsStopping(bool isStopping);
    bool GetIsStopped();
    int GetStereoFPS();

private:
    // Declare class objects.
    FPS*                        StereoFPSCounter;

    // Declare class variables.
    int                         StereoFPSCount;
    bool						isStopping;
    bool						isStopped;

    // Declare constants.
    const double PI                                     = 3.14159265358979323846;
};
///////////////////////////////////////////////////////////////////////////////
#endif