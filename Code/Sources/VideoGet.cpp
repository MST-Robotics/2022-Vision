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

    // Initialize Variables.
    camerasStarted                              = false;
    isStopping							        = false;
    isStopped							        = false;
    fpsCounters                                 = vector<FPS>(10);

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
    // Nothing to do for now.
}

/****************************************************************************
        Description:	Grabs frames from camera.

        Arguments: 		MAT&, MAT&, MAT&, BOOL&, BOOL&, BOOL&, VECTOR<CvSink>, SHARED_TIMED_MUTEX&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoGet::StartCapture(Mat &visionFrame, Mat &leftStereoFrame, Mat &rightStereoFrame, bool &cameraSourceIndex, bool &drivingMode, vector<CvSink> &cameraSinks, shared_timed_mutex &VisionMutex, shared_timed_mutex &StereoMutex)
{
    // Create a vector for storing created threads.
    vector<thread*> frameThreads;

    // Continuously manage threads and grab camera frames.
    while (1)
    {
        try
        {
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

                // Check if we already started cameras.
                if (!camerasStarted)
                {
                    // Loop through all connected cameras and store frames for the ones we want.
                    for (int i = 0; i < cameraSinks.size(); i++)
                    {
                        // If this camera has the correct alias for vision, use it as our vision camera.
                        if (cameraSinks[i].GetName().find(VISION_DASHBOARD_ALIAS) != string::npos)
                        {
                            // Check if this camera will also be used for left stereo vision.
                            if (cameraSinks[i].GetName().find(LEFT_STEREO_DASHBOARD_ALIAS) != string::npos)
                            {
                                // Grab camera frame for vision and left stereo.
                                // frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(visionFrame), ref(fpsCounters.at(0)), ref(isStopped), ref(VisionMutex)));
                                // frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(leftStereoFrame), ref(fpsCounters.at(1)), ref(isStopped), ref(StereoMutex)));
                                frameThreads.emplace_back(new thread(&VideoGet::GetTwoCameraFrames, this, ref(cameraSinks[i]), ref(visionFrame), ref(leftStereoFrame), ref(fpsCounters.at(0)), ref(fpsCounters.at(1)), ref(isStopped), ref(VisionMutex), ref(StereoMutex)));   
                            }
                            // Check if this camera will also be used for right stereo vision.
                            else if (cameraSinks[i].GetName().find(RIGHT_STEREO_DASHBOARD_ALIAS) != string::npos)
                            {
                                // Grab camera frame for vision.
                                // frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(visionFrame), ref(fpsCounters.at(0)), ref(isStopped), ref(VisionMutex)));
                                // frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(rightStereoFrame), ref(fpsCounters.at(2)), ref(isStopped), ref(StereoMutex)));
                                frameThreads.emplace_back(new thread(&VideoGet::GetTwoCameraFrames, this, ref(cameraSinks[i]), ref(visionFrame), ref(rightStereoFrame), ref(fpsCounters.at(0)), ref(fpsCounters.at(2)), ref(isStopped), ref(VisionMutex), ref(StereoMutex)));
                            }
                            else
                            {
                                // Grab camera frame for vision.
                                frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(visionFrame), ref(fpsCounters.at(0)), ref(isStopped), ref(VisionMutex)));
                            }
                        }
                        // If camera won't be used for vision or vision and stereo, then check for just left stereo.
                        else if (cameraSinks[i].GetName().find(LEFT_STEREO_DASHBOARD_ALIAS) != string::npos)
                        {
                            // Grab camera frame for vision.
                            frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(leftStereoFrame), ref(fpsCounters.at(1)), ref(isStopped), ref(StereoMutex)));
                        }
                        // If camera won't be used for vision or vision and stereo, then check for just right stereo.
                        else if (cameraSinks[i].GetName().find(RIGHT_STEREO_DASHBOARD_ALIAS) != string::npos)
                        {
                            // Grab camera frame for vision.
                            frameThreads.emplace_back(new thread(&VideoGet::GetCameraFrames, this, ref(cameraSinks[i]), ref(rightStereoFrame), ref(fpsCounters.at(2)), ref(isStopped), ref(StereoMutex)));
                        }
                    }

                    // Set toggle.
                    camerasStarted = true;
                }
            }
        }
        catch (const exception& e)
        {
            //SetIsStopping(true);
            cout << "WARNING: Video data empty or camera not present." << "\n" << e.what() << endl;
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
                        It continuously gets new camera frames and stores them in
                        given mat.

        Arguments: 		CvSink&, MAT&, FPS&, BOOL&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoGet::GetCameraFrames(CvSink &camera, Mat &mainFrame, FPS &fpsCounter, bool &stop, shared_timed_mutex &Mutex)
{
    // Loop forever.
    while (!stop)
    {
        // Acquire resource lock from process thread. This will block the process thread until processing is done.
        unique_lock<shared_timed_mutex> guard(Mutex);

        // Get camera frame.
        int retTime = camera.GrabFrame(mainFrame, FRAME_GET_TIMEOUT);

        // Increment FPS counter.
        fpsCounter.Increment();
    }
}

/****************************************************************************
        Description:	This is a container method for a thread and never exits.
                        It continuously gets new camera frames and stores them in
                        the two given mats. Seperate locks are aquired for each.

        Arguments: 		CvSink&, MAT&, MAT&, FPS&, BOOL&, SHARED_TIMED_MUTEX&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoGet::GetTwoCameraFrames(CvSink &camera, Mat &mainFrame, Mat &secondaryFrame, FPS &mainFPSCounter, FPS &secondaryFPSCounter, bool &stop, shared_timed_mutex &mainMutex, shared_timed_mutex &secondaryMutex)
{
    // Loop forever.
    while (!stop)
    {
        // Acquire resource lock from process thread. This will block the process thread until processing is done.
        // unique_lock<shared_timed_mutex> mainGuard(mainMutex);
        // Get camera frame.
        int retTime = camera.GrabFrame(mainFrame, FRAME_GET_TIMEOUT);

        // Acquire resource lock from process thread. This will block the process thread until processing is done.
        // unique_lock<shared_timed_mutex> secondaryGuard(secondaryMutex);
        // Copy new frames stored in mainFrame mat to secondaryFrame mat.
        secondaryFrame = mainFrame.clone();

        // Increment FPS counters.
        mainFPSCounter.Increment();
        secondaryFPSCounter.Increment();
    }
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
        Description:	Gets the current FPS of the indexed thread.

        Arguments: 		INT

        Returns: 		Int
****************************************************************************/
int VideoGet::GetFPS(const int index)
{
    return fpsCounters.at(index).FramesPerSec();
}
///////////////////////////////////////////////////////////////////////////////