/****************************************************************************
			Description:	Defines the VideoShow Class

			Classes:		VideoShow

			Project:		MATE 2022

			Copyright 2021 MST Design Team - Underwater Robotics.
****************************************************************************/
#ifndef VideoShow_h
#define VideoShow_h

#include <cstdio>
#include <string>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <iostream>
#include <algorithm>
#include <vector>

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

// Define constants.
const string VISION_PROCESSED_STREAM_ALIAS = "vision";
const string STEREO_PROCESSED_STREAM_ALIAS = "stereo";
///////////////////////////////////////////////////////////////////////////////

class VideoShow
{
public:
    // Define class methods.
    VideoShow();
    ~VideoShow();
    void ShowFrame(Mat &finalImg, Mat &stereoImg, vector<CvSource> &cameraSources, shared_timed_mutex &VisionMutex, shared_timed_mutex &StereoMutex);
    void SetIsStopping(bool isStopping);
    bool GetIsStopped();
    int GetFPS(const int index);

private:
    // Declare private class methods.
    void ShowCameraFrames(CvSource &source, Mat &mainFrame, FPS &fpsCounter, shared_timed_mutex &Mutex);

    // Declare class objects and variables.
    bool                    camerasStarted;  
    bool					isStopping;
    bool					isStopped;
    vector<FPS>             fpsCounters;
};
///////////////////////////////////////////////////////////////////////////////
#endif