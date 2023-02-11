/****************************************************************************
			Description:	Defines the VideoGet Class

			Classes:		VideoGet

			Project:		2020 DeepSpace Vision Code

			Copyright 2020 First Team 3284 - Camdenton LASER Robotics.
****************************************************************************/
#ifndef VideoGet_h
#define VideoGet_h

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
#include <cameraserver/CameraServer.h>

using namespace std;
using namespace cv;
using namespace cs;

// Define constants.
const bool USE_VIRTUAL_CAM = false;
const string VISION_DASHBOARD_ALIAS = "vision";
const string LEFT_STEREO_DASHBOARD_ALIAS = "left_stereo";
const string RIGHT_STEREO_DASHBOARD_ALIAS = "right_stereo";
const int FRAME_GET_TIMEOUT = 1.0;
///////////////////////////////////////////////////////////////////////////////

class VideoGet
{
public:
    // Declare class methods.
    VideoGet();
    ~VideoGet();
    void StartCapture(Mat &visionFrame, Mat &leftStereoFrame, Mat &rightStereoFrame, bool &cameraSourceIndex, bool &drivingMode, vector<CvSink> &cameraSinks, shared_timed_mutex &VisionMutex, shared_timed_mutex &StereoMutex);
    void SetIsStopping(bool isStopping);
    bool GetIsStopped();
    int GetFPS(const int index);

private:
    // Declare private class methods.
    void GetCameraFrames(CvSink &camera, Mat &mainFrame, FPS &fpsCounter, bool &stop, shared_timed_mutex &Mutex);

    // Declare class objects and variables.
    VideoCapture			cap;
    
    bool                    camerasStarted;
    bool					isStopping;
    bool					isStopped;
    vector<FPS>             fpsCounters;
};
///////////////////////////////////////////////////////////////////////////////
#endif
