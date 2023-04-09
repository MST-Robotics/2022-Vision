/****************************************************************************
			Description:	Defines the VideoProcess Class

			Classes:		VideoProcess

			Project:		MATE 2022

			Copyright 2021 MST Design Team - Underwater Robotics.
****************************************************************************/
#ifndef VideoProcess_h
#define VideoProcess_h

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
#include "ModelUtils.h"
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
#include <tensorflow/lite/interpreter.h>

using namespace cv;
using namespace cs;
using namespace nt;
using namespace frc;
using namespace wpi;
using namespace std;
///////////////////////////////////////////////////////////////////////////////


class VideoProcess
{
public:
    // Declare class methods.
    VideoProcess();
    ~VideoProcess();
    void Process(Mat &frame, Mat &finalImg, int &targetCenterX, int &targetCenterY, int &centerLineTolerance, double &contourAreaMinLimit, double &contourAreaMaxLimit, bool &tuningMode, bool &drivingMode, int &trackingMode, bool &takeShapshot, bool &solvePNPEnabled, vector<int> &trackbarValues, vector<double> &trackingResults, vector<double> &solvePNPValues, vector<string> &classList, cv::dnn::Net &onnxModel, unique_ptr<tflite::Interpreter> &tfliteModelInterpreter, float &neuralNetworkMinConfidence, bool &forceONNXModel, VideoGet &VideoGetter, shared_timed_mutex &MutexGet, shared_timed_mutex &MutexShow);
    vector<double> SolveObjectPose(vector<Point2f> imagePoints, Mat &finalImg, Mat &frame, int targetPositionX, int targetPositionY);
    int SignNum(double val);
    void SetIsStopping(bool isStopping);
    bool GetIsStopped();
    int GetVisionFPS();
    int GetStereoFPS();

    // Declare public variables.
    enum TrackingMode
    {
        TRENCH_TRACKING = 0,
        LINE_TRACKING,
        FISH_TRACKING,
        TAPE_TRACKING
    };

private:
    // Declare class objects.
    Mat							HSVImg;
    Mat							blurImg;
    Mat							filterImg;
    Mat							dilateImg;
    Mat							corners;
    Mat							cornersNormalized;
    Mat							cornersScaled;
    Mat							kernel;
    Mat							cameraMatrix;
    Mat							distanceCoefficients;
    vector<Point3f>				objectPoints;
    vector<vector<Point>>		contours;
    vector<Vec4i>				hierarchy;
    vector<vector<Scalar>>      colorRanges;
    vector<string>              colors;
    FPS*						VisionFPSCounter;

    // Declare class variables.
    int                         VisionFPSCount;
    bool						isStopping;
    bool						isStopped;

    // Declare constants.
    const Mat KERNEL								    = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    const int GREEN_BLUR_RADIUS						    = 3;
    const int HORIZONTAL_ASPECT						    = 4;
    const int VERTICAL_ASPECT						    = 3;
    const int CAMERA_FOV							    = 75;
    const int SCREEN_WIDTH							    = 640;
    const int SCREEN_HEIGHT							    = 480;
    const int DNN_MODEL_IMAGE_SIZE                      = 640;
    const double DNN_NMS_THRESH                         = 0.4;
    const double PI                                     = 3.14159265358979323846;
    const double FOCAL_LENGTH						    = (SCREEN_WIDTH / 2.0) / tan((CAMERA_FOV * PI / 180.0) / 2.0);
    const vector<Scalar> DETECTION_COLORS               = {Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0)};
};
///////////////////////////////////////////////////////////////////////////////
#endif