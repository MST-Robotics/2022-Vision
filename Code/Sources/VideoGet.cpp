/****************************************************************************
			Description:	Implements the VideoGet Class.

			Classes:		VideoGet

			Project:		MATE 2022

			Copyright 2021 MST Design Team - Underwater Robotics.
****************************************************************************/
#include "../Headers/VideoGet.h"
///////////////////////////////////////////////////////////////////////////////


/****************************************************************************
            Description:	VideoGet constructor.

            Arguments:		None

            Derived From:	Nothing
****************************************************************************/
VideoGet::VideoGet()
{
    // Create objects.
    FPSCounter									= new FPS();

    // Initialize Variables.
    isStopping							= false;
    isStopped							= false;

    // Create VideoCapture object for reading video from virtual cam if enabled.
    if (USE_VIRTUAL_CAM)
    {
        cap = VideoCapture();
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FPS, 30);
        cap.open(0);
    }
}

/****************************************************************************
        Description:	VideoGet destructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
VideoGet::~VideoGet()
{
    // Delete object pointers.
    delete FPSCounter;

    // Set object pointers as nullptrs.
    FPSCounter				 = nullptr;
}

/****************************************************************************
        Description:	Grabs frames from camera.

        Arguments: 		MAT&, MAT&, MAT&, BOOL&, BOOL&, BOOL&, VECTOR<CvSink>, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoGet::StartCapture(Mat &visionFrame, Mat &leftStereoFrame, Mat &rightStereoFrame, bool &cameraSourceIndex, bool &drivingMode, vector<CvSink> &cameraSinks, shared_timed_mutex &Mutex)
{
    // Continuously grab camera frames.
    while (1)
    {
        // Increment FPS counter.
        FPSCounter->Increment();

        try
        {
            // Acquire resource lock for thread.
            lock_guard<shared_timed_mutex> guard(Mutex);		// unique_lock

            // Check if we are using virtual camera.
            if (USE_VIRTUAL_CAM)
            {
                // Read frame from video file.
                cap >> visionFrame;
            }
            else
            {
                // If the frame is empty, stop the capture.
                if (cameraSinks.empty())
                {
                    break;
                }

                cameraSinks[1].GrabFrame(rightStereoFrame);
                // Loop through all connected cameras and store frames for the ones we want.
                for (int i = 0; i < cameraSinks.size(); i++)
                {
                    // If this camera has the correct alias for vision, use it as our vision camera.
                    if (cameraSinks[i].GetName().find(VISION_DASHBOARD_ALIAS) != string::npos)
                    {
                        // Grab camera frame for vision.
                        cameraSinks[i].GrabFrame(visionFrame);

                        // Check if this camera will also be used for left stereo vision.
                        if (cameraSinks[i].GetName().find(LEFT_STEREO_DASHBOARD_ALIAS) != string::npos)
                        {
                            // Stored already grabbed frame into left frame.
                            leftStereoFrame = visionFrame.clone();
                        }
                        // Check if this camera will also be used for right stereo vision.
                        else if (cameraSinks[i].GetName().find(RIGHT_STEREO_DASHBOARD_ALIAS) != string::npos)
                        {
                            // Stored already grabbed frame into left frame.
                            rightStereoFrame = visionFrame.clone();
                        }
                    }
                    // If camera won't be used for vision or vision and stereo, then check for just left stereo.
                    else if (cameraSinks[i].GetName().find(LEFT_STEREO_DASHBOARD_ALIAS) != string::npos)
                    {
                        // Grab camera frame for left stereo image.
                        cameraSinks[i].GrabFrame(leftStereoFrame);
                    }
                    // If camera won't be used for vision or vision and stereo, then check for just right stereo.
                    else if (cameraSinks[i].GetName().find(RIGHT_STEREO_DASHBOARD_ALIAS) != string::npos)
                    {
                        // Grab camera frame for left stereo image.
                        cameraSinks[i].GrabFrame(rightStereoFrame);
                    }
                }
            }
        }
        catch (const exception& e)
        {
            //SetIsStopping(true);
            cout << "WARNING: Video data empty or camera not present." << "\n" << e.what() << endl;
        }

        // Calculate FPS.
        FPSCount = FPSCounter->FramesPerSec();

        // If the program stops shutdown the thread.
        if (isStopping)
        {
            break;
        }
    }

    // Clean-up.
    isStopped = true;
    return;
}

/****************************************************************************
        Description:	Signals the thread to stop.

        Arguments: 		BOOL

        Returns: 		Nothing
****************************************************************************/
void VideoGet::SetIsStopping(bool isStopping)
{
    this->isStopping = isStopping;
}

/****************************************************************************
        Description:	Gets if the thread has stopped.

        Arguments: 		None

        Returns: 		BOOL
****************************************************************************/
bool VideoGet::GetIsStopped()
{
    return isStopped;
}

/****************************************************************************
        Description:	Gets the current FPS of the thread.

        Arguments: 		None

        Returns: 		Int
****************************************************************************/
int VideoGet::GetFPS()
{
    return FPSCount;
}
///////////////////////////////////////////////////////////////////////////////