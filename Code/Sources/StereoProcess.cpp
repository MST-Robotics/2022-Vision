/****************************************************************************
			Description:	Implements the StereoProcess Class

			Classes:		StereoProcess

			Project:		MATE 2022

			Copyright 2022 MST Design Team - Underwater Robotics.
****************************************************************************/
#include "../Headers/StereoProcess.h"
///////////////////////////////////////////////////////////////////////////////


/****************************************************************************
        Description:	StereoProcess constructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
StereoProcess::StereoProcess()
{
    // Create object pointers.
    StereoFPSCounter                        = new FPS();
    
    // Initialize member variables.
    StereoFPSCount                          = 0;
    isStopping							    = false;
    isStopped							    = false;
}

/****************************************************************************
        Description:	StereoProcess destructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
StereoProcess::~StereoProcess()
{
    // Delete object pointers.
    delete StereoFPSCounter;

    // Set object pointers as nullptrs.
    StereoFPSCounter = nullptr;
}

/****************************************************************************
        Description:	Processes two frames and finds the disparity map between them for depth estimation.

        Arguments:      MAT&, MAT&, MAT&, BOOL&, SHARED_TIMED_MUTEX&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void StereoProcess::Process(Mat &leftFrame, Mat &rightFrame, Mat &stereoImg, bool &enableStereoVision, VideoGet &VideoGetter, shared_timed_mutex &MutexGet, shared_timed_mutex &MutexShow)
{
    // Give other threads enough time to start before processing camera frames.
    this_thread::sleep_for(std::chrono::milliseconds(800));

    // Processing loop for stereo vision.
    while (1)
    {
        // Increment FPS Counter.
        StereoFPSCounter->Increment();

        // Acquire resource lock for show thread only after frame has been used.
        unique_lock<shared_timed_mutex> guard(MutexShow);

        // Catch corrupt frames.
        try
        {
            // Acquire read-only resource lock for get thread. If the other thead is writing, then this will block until it's done.
            shared_lock<shared_timed_mutex> guard(MutexGet);

            // Don't compute if either frames are empty.
            if (!leftFrame.empty() || !rightFrame.empty())
            {
                // Check if stereo vision computing is enabled.
                if (enableStereoVision)
                {
                    stereoImg = leftFrame.clone();
                }
                else
                {
                    // Set existing image to black.
                    stereoImg.setTo(Scalar(5, 10, 15));
                    // Overlay text telling user stereo computation is disabled.
                    putText(stereoImg, "Stereo computation is disabled by default to", Point(stereoImg.cols / 12, stereoImg.rows / 4), FONT_HERSHEY_DUPLEX, 0.65, Scalar(250, 100, 100), 1);
                    putText(stereoImg, "save resources. Enabled it through the shuffleboard.", Point(stereoImg.cols / 12, stereoImg.rows / 3), FONT_HERSHEY_DUPLEX, 0.65, Scalar(250, 100, 100), 1);
                }

                // Put FPS on image.
                StereoFPSCount = StereoFPSCounter->FramesPerSec();
                putText(stereoImg, ("Left Camera FPS: " + to_string(VideoGetter.GetFPS(1))), Point(400, stereoImg.rows - 60), FONT_HERSHEY_DUPLEX, 0.65, Scalar(200, 200, 200), 1);
                putText(stereoImg, ("Right Camera FPS: " + to_string(VideoGetter.GetFPS(2))), Point(400, stereoImg.rows - 40), FONT_HERSHEY_DUPLEX, 0.65, Scalar(200, 200, 200), 1);
                putText(stereoImg, ("Stereo FPS: " + to_string(StereoFPSCount)), Point(450, stereoImg.rows - 20), FONT_HERSHEY_DUPLEX, 0.65, Scalar(200, 200, 200), 1);
            }
        }
        catch (const exception& e)
        {
            //SetIsStopping(true);
            // Print error to console and show that an error has occured on the screen.
            putText(stereoImg, "Stereo image Processing ERROR", Point(280, stereoImg.rows - 440), FONT_HERSHEY_DUPLEX, 0.65, Scalar(0, 0, 250), 1);
            cout << "\nWARNING: MAT corrupt or a runtime error has occured! Frame has been dropped." << "\n" << e.what() << endl;
        }

        // If the program stops shutdown the thread.
        if (isStopping)
        {
            break;
        } 
    }
}

/****************************************************************************
        Description:	Signals the thread to stop.

        Arguments: 		BOOL

        Returns: 		Nothing
****************************************************************************/
void StereoProcess::SetIsStopping(bool isStopping)
{
    this->isStopping = isStopping;
}

/****************************************************************************
        Description:	Gets if the thread has stopped.

        Arguments: 		None

        Returns: 		BOOL
****************************************************************************/
bool StereoProcess::GetIsStopped()
{
    return isStopped;
}

/****************************************************************************
        Description:	Gets the current FPS of the stereo thread.

        Arguments: 		None

        Returns: 		INT
****************************************************************************/
int StereoProcess::GetStereoFPS()
{
    return StereoFPSCount;
}