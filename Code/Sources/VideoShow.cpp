/****************************************************************************
			Description:	Implements the VideoShow Class

			Classes:		VideoShow

			Project:		MATE 2022

			Copyright 2021 MST Design Team - Underwater Robotics.
****************************************************************************/
#include "../Headers/VideoShow.h"
///////////////////////////////////////////////////////////////////////////////


/****************************************************************************
        Description:	VideoShow constructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
VideoShow::VideoShow()
{
    // Create objects.
    fpsCounters                         = vector<FPS>(10);

    // Initialize member variables.
    camerasStarted                      = false;
    isStopping							= false;
    isStopped							= false;
}

/****************************************************************************
        Description:	VideoShow destructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
VideoShow::~VideoShow()
{
    // Nothing to do.
}

/****************************************************************************
        Description:	Method that gives the processed frame to CameraServer.

        Arguments: 		MAT&, MAT&, VECTOR<CVSOURCE>&, SHARED_TIMED_MUTEX&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoShow::ShowFrame(Mat &finalImg, Mat &stereoImg, vector<CvSource> &cameraSources, shared_timed_mutex &VisionMutex, shared_timed_mutex &StereoMutex)
{
    // Give other threads some time.
    this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Create a vector for storing created threads.
    vector<thread*> frameThreads;

    // Continuously manage threads and show frames.
    while (1)
    {
        // Check to make sure frame is not corrupt.
        try
        {
            // If the frame is empty, stop the capture.
            if (cameraSources.empty())
            {
                break;
            }

            // Check if we already started cameras.
            if (!camerasStarted)
            {
                // Loop through all create camera streams and send frames to each one.
                for (int i = 0; i < cameraSources.size(); i++)
                {
                    // If this camera has the correct alias for vision, use it as our vision stream.
                    if (cameraSources[i].GetName().find(VISION_PROCESSED_STREAM_ALIAS) != string::npos)
                    {
                        // Put camera frame for vision.
                        frameThreads.emplace_back(new thread(&VideoShow::ShowCameraFrames, this, ref(cameraSources[i]), ref(finalImg), ref(fpsCounters.at(0)), ref(VisionMutex)));
                    }
                    // Check if this source will be used for stereo vision stream, but make sure it doesn't contain the vision alias.
                    else if (cameraSources[i].GetName().find(STEREO_PROCESSED_STREAM_ALIAS) != string::npos && cameraSources[i].GetName().find(VISION_PROCESSED_STREAM_ALIAS) == string::npos)
                    {
                        // Put camera frame for stereo.
                        frameThreads.emplace_back(new thread(&VideoShow::ShowCameraFrames, this, ref(cameraSources[i]), ref(stereoImg), ref(fpsCounters.at(1)), ref(StereoMutex)));
                    }
                }

                // Set toggle.
                camerasStarted = true;
            }
        }
        catch (const exception& e)
        {
            //SetIsStopping(true);
            cout << "WARNING: MAT corrupt. Frame has been dropped." << endl;
        }

        // If the program stops shutdown the thread.
        if (isStopping)
        {
            break;
        }

        // Sleep to save CPU time. Thead management doesn't need to update very fast.
        this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Loop through the threads in the threads vector and join them.
    for (thread* task : frameThreads)
    {
        // Join.
        task->join();

        // Delete dynamically allocated object.
        delete task;
        // Set old pointer to null.
        task = nullptr;
    }

    // Clean-up.
    isStopped = true;
}

/****************************************************************************
        Description:	This is a container method for a thread and never exits.
                        It continuously shows new camera frames.

        Arguments: 		CvSource&, MAT&, FPS&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoShow::ShowCameraFrames(CvSource &source, Mat &mainFrame, FPS &fpsCounter, shared_timed_mutex &Mutex)
{
    // Loop forever.
    while (!isStopping)
    {
        // Acquire resource lock from process thread. This will block the process thread until processing is done.
        shared_lock<shared_timed_mutex> guard(Mutex);

        // Output frame to camera stream.
        source.PutFrame(mainFrame);

        // Increment FPS counter.
        fpsCounter.Increment();
    }
}

/****************************************************************************
        Description:	Signals the thread to stop.

        Arguments: 		BOOL

        Returns: 		Nothing
****************************************************************************/
void VideoShow::SetIsStopping(bool isStopping)
{
    this->isStopping = isStopping;
}

/****************************************************************************
        Description:	Gets if the thread has stopped.

        Arguments: 		Nothing

        Returns: 		BOOL 
****************************************************************************/
bool VideoShow::GetIsStopped()
{
    return isStopped;
}

/****************************************************************************
        Description:	Gets the current FPS of the thread.

        Arguments: 		CONST INT

        Returns: 		INT
****************************************************************************/
int VideoShow::GetFPS(const int index)
{
    return fpsCounters.at(index).FramesPerSec();
}
///////////////////////////////////////////////////////////////////////////////