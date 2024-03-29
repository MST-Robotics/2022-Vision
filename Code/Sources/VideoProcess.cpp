/****************************************************************************
			Description:	Implements the VideoProcess Class

			Classes:		VideoProcess

			Project:		MATE 2022

			Copyright 2021 MST Design Team - Underwater Robotics.
****************************************************************************/
#include "../Headers/VideoProcess.h"
///////////////////////////////////////////////////////////////////////////////


/****************************************************************************
        Description:	VideoProcess constructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
VideoProcess::VideoProcess()
{
    // Create object pointers.
    FPSCounter							    = new FPS();
    
    // Initialize member variables.
    FPSCount                                = 0;
    isStopping							    = false;
    isStopped							    = false;

    // Setup colors and ranges for box tape detection. (lowerthresh, upperthresh, tracking overlay color(B,G,R))
    colorRanges.emplace_back(vector<Scalar> { Scalar(91, 219, 118), Scalar(255, 255, 157), Scalar(255, 156, 64) });         // lightblue
    colorRanges.emplace_back(vector<Scalar> { Scalar(100, 230, 45), Scalar(255, 255, 95), Scalar(219, 4, 12) });            // blue
    colorRanges.emplace_back(vector<Scalar> { Scalar(0, 228, 90), Scalar(68, 255, 163), Scalar(0, 242, 255) });             // yellow
    colorRanges.emplace_back(vector<Scalar> { Scalar(57, 230, 58), Scalar(71, 255, 211), Scalar(11, 117, 25) });            // green
    colorRanges.emplace_back(vector<Scalar> { Scalar(128, 70, 0), Scalar(255, 201, 60), Scalar(255, 0, 195) });             // purple
    colorRanges.emplace_back(vector<Scalar> { Scalar(0, 177, 15), Scalar(61, 255, 90), Scalar(9, 112, 222) });              // orange
    colors.emplace_back("lightblue");
    colors.emplace_back("blue");
    colors.emplace_back("yellow");
    colors.emplace_back("green");
    colors.emplace_back("purple");
    colors.emplace_back("orange");

    ////
    // Setup SolvePNP data.
    ////
    // Reference object points.
    objectPoints.emplace_back(Point3f(39.50, 0.0, 0.0));
    objectPoints.emplace_back(Point3f(29.50, -17.0, 0.0));
    objectPoints.emplace_back(Point3f(9.75, -17.0, 0.0));
    objectPoints.emplace_back(Point3f(0.0, 0.0, 0.0));

    // Precalibrated camera matrix values.
    double mtx[3][3] = {{516.5613698781304, 0.0, 320.38297194779585},		//// PSEye Cam
                        {0.0, 515.9356734667019, 231.73585601568368},
                        {0.0, 0.0, 1.0}};
    // double mtx[3][3] = {{659.3851992714341, 0.0, 306.98918779442675},		//// Lifecam
    // 					{0.0, 659.212123568372, 232.07157473243464},
    // 					{0.0, 0.0, 1.0}};
    cameraMatrix = Mat(3, 3, CV_64FC1, mtx).clone();

    // Precalibration distance/distortion values.
    double dist[5] = {-0.0841024904469607, 0.014864043816324026, -0.00013887041018197853, -0.0014661216967276468, 0.5671907234987197};	//// PSEye Cam
    // double dist[5] = {0.1715327237204972, -1.3255106761114646, 7.713495040297368e-07, -0.0035865453000784634, 2.599132082766894};	//// Lifecam
    distanceCoefficients = Mat(1, 5, CV_64FC1, dist).clone();
}

/****************************************************************************
        Description:	VideoProcess destructor.

        Arguments:		None

        Derived From:	Nothing
****************************************************************************/
VideoProcess::~VideoProcess()
{
    // Delete object pointers.
    delete FPSCounter;

    // Set object pointers as nullptrs.
    FPSCounter = nullptr;
}

/****************************************************************************
        Description:	Processes frames with OpenCV.

        Arguments(dear god help us): MAT&, MAT&, INT&, INT&, DOUBLE&, DOUBLE&, BOOL&, BOOL&, BOOL&, BOOL&, VECTOR<INT>, VECTOR<DOUBLE>, VIDEOGET, SHARED_TIMED_MUTEX&, SHARED_TIMED_MUTEX&

        Returns: 		Nothing
****************************************************************************/
void VideoProcess::Process(Mat &frame, Mat &finalImg, int &targetCenterX, int &targetCenterY, int &centerLineTolerance, double &contourAreaMinLimit, double &contourAreaMaxLimit, bool &tuningMode, bool &drivingMode, int &trackingMode, bool &takeShapshot, bool &solvePNPEnabled, vector<int> &trackbarValues, vector<double> &trackingResults, vector<double> &solvePNPValues, vector<string> &classList, cv::dnn::Net &onnxModel, VideoGet &VideoGetter, shared_timed_mutex &MutexGet, shared_timed_mutex &MutexShow)
{
    // Give other threads enough time to start before processing camera frames.
    this_thread::sleep_for(std::chrono::milliseconds(800));

    while (1)
    {
        // Increment FPS counter.
        FPSCounter->Increment();

        // Make sure frame is not corrupt.
        try
        {
            // Acquire resource lock for read thread. NOTE: This line has been commented out to improve processing speed. VideoGet takes to long with the resources.
            // shared_lock<shared_timed_mutex> guard(MutexGet);

            if (!frame.empty())
            {
                // Acquire resource lock for show thread only after frame has been used.
                unique_lock<shared_timed_mutex> guard(MutexShow);
                // Copy frame to a new mat.
                finalImg = frame.clone();
                
                // Reset tracking results array every iteration.
                trackingResults.clear();

                // Driving mode.
                if (!drivingMode)
                {
                    // Tracking mode. (TrackingMode enum)
                    switch (trackingMode)
                    {
                        /****************************************************
                        *			Track trench target
                        *****************************************************/
                        case TRENCH_TRACKING:
                        {
                            // Convert image from RGB to HSV.
                            cvtColor(frame, HSVImg, COLOR_BGR2HSV);
                            // Blur the image.
                            blur(HSVImg, blurImg, Size(GREEN_BLUR_RADIUS, GREEN_BLUR_RADIUS));
                            // Filter out specific color in image.
                            inRange(blurImg, Scalar(trackbarValues[0], trackbarValues[2], trackbarValues[4]), Scalar(trackbarValues[1], trackbarValues[3], trackbarValues[5]), filterImg);
                            // Remove small blobs.
                            erode(filterImg, dilateImg, KERNEL);
                            // "Inflate" image.
                            dilate(dilateImg, dilateImg, KERNEL);

                            // Find countours of image.
                            findContours(dilateImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);		////RETR_TREE //// TRY CHAIN_APPROX_SIMPLE		//// Not sure what this method of detection does, but it worked before: CHAIN_APPROX_TC89_KCOS

                            // Draw all contours in white.
                            // drawContours(finalImg, contours, -1, Scalar(255, 255, 210), 1, LINE_4, hierarchy);

                            // Only continue if we have more than two contours.
                            if (contours.size() >= 2)
                            {
                                // 'Round off' all contours with convexHull.
                                vector<vector<Point>> hulls;
                                for (vector<Point> contour : contours)
                                {
                                    vector<Point> hull;
                                    convexHull(contour, hull);
                                    hulls.emplace_back(hull);
                                }
                                
                                // Sort contours from biggest to smallest.
                                sort(hulls.begin(), hulls.end(), [](const vector<Point>& c1, const vector<Point>& c2) {	return fabs(contourArea(c1, false)) > fabs(contourArea(c2, false)); });
                                
                                // Remove contours whose area doesn't meet the threshold.
                                vector<vector<Point>> filteredHulls;
                                for (vector<Point> hull : hulls)
                                {
                                    double area = contourArea(hull);
                                    if (area >= contourAreaMinLimit && area <= contourAreaMaxLimit)
                                    {
                                        filteredHulls.emplace_back(hull);
                                    }
                                }

                                // Only continue if we have more than two contours.
                                if (filteredHulls.size() > 2)
                                {
                                    // Draw convex hull contours.
                                    polylines(finalImg, filteredHulls, true, Scalar(255, 255, 210), 1);

                                    // Store the upper and lower extremes of each hull contour.
                                    vector<vector<int>> hullExtremes;
                                    for (vector<Point> hull : filteredHulls)
                                    {
                                        // Find and store the bounding rect. (Rect type contains x, y, height, width)
                                        auto val = minmax_element(hull.begin(), hull.end(), [](Point const& a, Point const& b) { return a.y < b.y; });
                                        vector<int> point;
                                        point.emplace_back(val.first->x);
                                        point.emplace_back(val.first->y);
                                        point.emplace_back(val.second->x);
                                        point.emplace_back(val.second->y);
                                        hullExtremes.emplace_back(point);
                                    }

                                    // Now that we have the lines, find the tallest one.
                                    int minLineLength = 50;
                                    vector<int> tallestLine1 = {0, 0, 0, minLineLength};
                                    for (vector<int> line : hullExtremes)
                                    {
                                        // Compare the y distance of the line to the currently stored biggest one.
                                        if ((line[3] - line[1]) > (tallestLine1[3] - tallestLine1[1]))
                                        {
                                            tallestLine1.assign(line.begin(), line.end());
                                        }
                                    }

                                    // Remove the line we just found.
                                    hullExtremes.erase(remove(hullExtremes.begin(), hullExtremes.end(), tallestLine1));
                                    // Find the next tallest line segment.
                                    vector<int> tallestLine2 = {SCREEN_WIDTH, 0, SCREEN_WIDTH, minLineLength};
                                    for (vector<int> line : hullExtremes)
                                    {
                                        // Compare the y distance of the line to the currently stored biggest one.
                                        if ((line[3] - line[1]) > (tallestLine2[3] - tallestLine2[1]))
                                        {
                                            tallestLine2.assign(line.begin(), line.end());
                                        }
                                    }

                                    // Find the center line.
                                    vector<int> centerLine;
                                    if (tallestLine1[0] < tallestLine2[0])
                                    {
                                        centerLine = {(tallestLine1[0] + ((tallestLine2[0] - tallestLine1[0]) / 2)), tallestLine1[1], (tallestLine1[2] + ((tallestLine2[2] - tallestLine1[2]) / 2)), tallestLine1[3]};
                                    }
                                    else
                                    {
                                        centerLine = {(tallestLine2[0] + ((tallestLine1[0] - tallestLine2[0]) / 2)), tallestLine1[1], (tallestLine2[2] + ((tallestLine1[2] - tallestLine2[2]) / 2)), tallestLine1[3]};
                                    }

                                    // Calculate the X center of the center line.
                                    int lineCenterX = (((centerLine[0] - centerLine[2]) / 2) + centerLine[2]) - (SCREEN_WIDTH / 2);
                                    // Calculate the width of the pipe channel.
                                    int lineCenterY = fabs(((tallestLine1[0] - tallestLine1[2]) / 2) - ((tallestLine2[0] - tallestLine2[2]) / 2));
                                    // If center line is not close to the center of the screen, then don't draw and output zero.
                                    if (fabs(lineCenterX) < centerLineTolerance)
                                    {
                                        // Draw the two tallest line segments and the center line.
                                        line(finalImg, Point(tallestLine2[0], tallestLine2[1]), Point(tallestLine2[2], tallestLine2[3]), Scalar(255, 0, 0), 3, LINE_4, 0);
                                        line(finalImg, Point(tallestLine1[0], tallestLine1[1]), Point(tallestLine1[2], tallestLine1[3]), Scalar(255, 0, 0), 3, LINE_4, 0);
                                        line(finalImg, Point(centerLine[0], centerLine[1]), Point(centerLine[2], centerLine[3]), Scalar(0, 200, 0), 3, LINE_4, 0);
                                        
                                        // Push position of tracked target.
                                        targetCenterX = lineCenterX;
                                        targetCenterY = lineCenterY;
                                    }
                                    else
                                    {
                                        // Push a default center values.
                                        targetCenterX = 0;
                                        targetCenterY = -1;
                                    }

                                    // // Store/convert the hulls contours into a Mat.
                                    // Mat mEdgeImg = Mat::zeros(finalImg.size(), CV_8UC1);
                                    // polylines(mEdgeImg, hulls, true, Scalar(255, 255, 255), 8);
                                    // mEdgeImg.copyTo(dilateImg);
                                    // // drawContours(mEdgeImg, hulls, -1, Scalar(255, 255, 255), 1, LINE_4);

                                    // // Setup HoughLinesP function variables.
                                    // double dRHO = 1;									// Distance resolution in pixels of the hough grid.
                                    // double dTheta = PI / 30;							// Angular resolution in radians of the hough grid.
                                    // int nThreshold = 30;								// Minimum number of votes.
                                    // double dMinLineLength = 50;							// Minimum number of pixels making up a line.
                                    // double dMaxLineGap = 50;							// Maximum gap in pixels between connectable line segments.
                                    // // Use HoughLinesP algorithm to detect potential line segments.
                                    // vector<Vec4i> lines;
                                    // HoughLinesP(mEdgeImg, lines, dRHO, dTheta, nThreshold, dMinLineLength, dMaxLineGap);

                                    // // Draw the detected lines.
                                    // for (Vec4i line : lines)
                                    // {
                                    // 	// Draw line.
                                    // 	line(finalImg, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 4, LINE_4, 0);
                                    // }
                                    
                                    // Sort array based on coordinates (leftmost to rightmost) to make sure contours are adjacent.
                                    // sort(vBiggestContours.begin(), vBiggestContours.end(), [](const vector<double>& points1, const vector<double>& points2) { return points1[0] < points2[0]; }); 		// Sorts using nCX location.	
                                }
                            }
                            else
                            {
                                // No contours to track. Output zero.
                                targetCenterX = 0;
                                targetCenterY = -1;
                            }
                            break;
                        }    

                        /****************************************************
                        *			Track fish net line target
                        *****************************************************/
                        case LINE_TRACKING:
                        {
                            // Create instance variables.
                            int numberOfVerticalSplits = 8;
                            int numberOfHorizontalSplits = 8;
                            int splitSize = 0;
                            int oppositeScreenRes = 0;
                            static bool screenSplitToggle = false;
                            vector<Point> linePoints;

                            // Convert image from RGB to HSV.
                            cvtColor(frame, HSVImg, COLOR_BGR2HSV);
                            // Blur the image.
                            blur(HSVImg, blurImg, Size(GREEN_BLUR_RADIUS, GREEN_BLUR_RADIUS));
                            // Filter out specific color in image.
                            inRange(blurImg, Scalar(trackbarValues[0], trackbarValues[2], trackbarValues[4]), Scalar(trackbarValues[1], trackbarValues[3], trackbarValues[5]), filterImg);
                            // Remove small blobs.
                            erode(filterImg, dilateImg, KERNEL);
                            // "Inflate" image.
                            dilate(dilateImg, dilateImg, KERNEL);
                            
                            // Determine whether we are looking at a vertical or horizontal line.
                            vector<Mat> splitImages;
                            if (screenSplitToggle)
                            {
                                // Set splitSize for vertical screen.
                                splitSize = SCREEN_HEIGHT / numberOfVerticalSplits;
                                oppositeScreenRes = SCREEN_WIDTH;

                                // Split image vertically into rectangles.
                                for (int i = 1; i <= numberOfVerticalSplits; i++)
                                {
                                    // Create area template for cropping.
                                    Rect ROI(0, (splitSize * (i - 1)), oppositeScreenRes, splitSize);
                                    // Crop image.
                                    splitImages.emplace_back(dilateImg(ROI));
                                }
                            }
                            else
                            {
                                // Set splitSize for horizontal screen.
                                splitSize = SCREEN_WIDTH / numberOfHorizontalSplits;
                                oppositeScreenRes = SCREEN_HEIGHT;

                                // Split image horizontally into rectangles.
                                for (int i = 1; i <= numberOfHorizontalSplits; i++)
                                {
                                    // Create area template for cropping.
                                    Rect ROI((splitSize * (i - 1)), 0, splitSize, oppositeScreenRes);
                                    // Crop image.
                                    splitImages.emplace_back(dilateImg(ROI));
                                }
                            }

                            // Loop through split images, and find the biggest contours center point.
                            for (int i = 0; i < splitImages.size(); i++)
                            {
                                // Find countours of image.
                                findContours(splitImages[i], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                                
                                // 'Round off' all contours with convexHull.
                                // vector<vector<Point>> hulls;
                                // for (vector<Point> contour : contours)
                                // {
                                //     vector<Point> hull;
                                //     convexHull(contour, hull);
                                //     hulls.emplace_back(hull);
                                // }

                                // Find the biggest contour.
                                int biggestArea = contourAreaMinLimit;
                                vector<Point> biggestContour;
                                for (vector<Point> contour : contours)
                                {
                                    // Get current contour area.
                                    int area = contourArea(contour);
                                    // If bigger than last one, store it.
                                    if (area > biggestArea)
                                    {
                                        // Set new biggest area.
                                        biggestArea = area;
                                        // Store new biggest contour.
                                        biggestContour = contour;
                                    }
                                }

                                if (!biggestContour.empty())
                                {
                                    // Find the center point of biggest contour.
                                    Moments moment = moments(biggestContour, true);
                                    Point center(moment.m10 / moment.m00, moment.m01 / moment.m00);
                                    
                                    // Draw locations are different depending on whether we are splitting vertically or horizontally.
                                    if (screenSplitToggle)
                                    {
                                        // Check if current circle is close enough to last point before appending.
                                        if (linePoints.empty() || fabs(center.x - linePoints[linePoints.size() - 1].x) < contourAreaMaxLimit)
                                        {
                                            // Append center circle to array.
                                            linePoints.emplace_back(Point(center.x, (center.y + (splitSize * i))));

                                            // Draw contour outline and center onto image.
                                            polylines(finalImg(Rect(0, (splitSize * i), oppositeScreenRes, splitSize)), biggestContour, true, Scalar(50, 200, 50), 3); 
                                            circle(finalImg(Rect(0, (splitSize * i), oppositeScreenRes, splitSize)), center, 4, Scalar(255, 255, 255), 5);
                                        }
                                    }
                                    else
                                    {
                                        // Check if current circle is close enough to last point before appending.
                                        if (linePoints.empty() || fabs(center.y - linePoints[linePoints.size() - 1].y) < contourAreaMaxLimit)
                                        {
                                            // Append center circle to array.
                                            linePoints.emplace_back(Point((center.x + (splitSize * i)), center.y));

                                             // Draw contour outline and center onto image.
                                            polylines(finalImg(Rect((splitSize * i), 0, splitSize, oppositeScreenRes)), biggestContour, true, Scalar(50, 200, 50), 3); 
                                            circle(finalImg(Rect((splitSize * i), 0, splitSize, oppositeScreenRes)), center, 4, Scalar(255, 255, 255), 5);
                                        }
                                    }
                                }
                            }

                            // Draw a line between each circle.
                            for (int i = 1; i < linePoints.size(); i++)
                            {
                                // Draw.
                                line(finalImg, linePoints[i - 1], linePoints[i], Scalar(255, 0, 0), LINE_4);
                            }

                            // Send line tracking data to main thread if not empty.
                            if (!linePoints.empty())
                            {
                                // Send whether line is vertical is horizontal.
                                trackingResults.emplace_back(screenSplitToggle);

                                // Send data point data.
                                for (Point point : linePoints)
                                {
                                    // Append x, y data in pairs in sequence.
                                    trackingResults.emplace_back(point.x);
                                    trackingResults.emplace_back(point.y);
                                }
                            }

                            // Flip-flop between vertical or horizontal splitting if our detected circles is low.
                            if (linePoints.size() < 3)
                            {
                                screenSplitToggle = !screenSplitToggle;
                            }
                            break;
                        }

                        /****************************************************
                        *	Track dead and alive fish with YOLO neural network
                        *****************************************************/
                        case FISH_TRACKING:
                        {
                            // Calculate the frame width, length, and max size.
                            int frameWidth = frame.cols;
                            int frameHeight = frame.rows;
                            int maxRes = MAX(frameWidth, frameHeight);
                            // Make a new square mat with the masRes size.
                            Mat resized = Mat::zeros(maxRes, maxRes, CV_8UC3);
                            // Copy the camera image into the new resized mat.
                            frame.copyTo(resized(Rect(0, 0, frameWidth, frameHeight)));
                            // resize(frame, frame, Size(DNN_MODEL_IMAGE_SIZE, DNN_MODEL_IMAGE_SIZE));
                            
                            // Resize to 640x640, normalize to [0,1] and swap red and blue channels. This creates a 4D blob from the image.
                            Mat result;
                            cv::dnn::blobFromImage(frame, result, 1.0 / 255.0, Size(DNN_MODEL_IMAGE_SIZE, DNN_MODEL_IMAGE_SIZE), Scalar(), true, false);
                            // Set the model's current input image.
                            onnxModel.setInput(result);

                            // Forward image through model layers and get the resulting predictions. This is the heavy comp shit.
                            vector<Mat> predictions;
                            onnxModel.forward(predictions, onnxModel.getUnconnectedOutLayersNames());
                            // const Mat &outputs = predictions[0];
                            
                            // Get image and model width and height ratios.
                            double widthFactor = double(frameWidth) / DNN_MODEL_IMAGE_SIZE;
                            double heightFactor = double(frameHeight) / DNN_MODEL_IMAGE_SIZE;
                            // Get class and detection data from output result.
                            float *data = (float*)predictions[0].data;
                            // Create instance variables for storing data while looping through detections.
                            vector<int> classIDs;
                            vector<float> confidences;
                            vector<Rect> predictionBoxes;
                            // Loop through each prediction. This array has 25,200 positions where each position is upto 85-length 1D array. 
                            // Each 1D array holds the data of one detection. The 4 first positions of this array are the xywh coordinates 
                            // of the bound box rectangle. The fifth position is the confidence level of that detection. The 6th up to 85th 
                            // elements are the scores of each class. For COCO with 80 classes outputs will be shape(n,85) with 
                            // 85 dimension = (x,y,w,h,object_conf, class0_conf, class1_conf, ...)
                            // I'm using ++i because it actually avoids a copy every iteration.
                            for (int i = 0; i < 25200; ++i) {
                                // Get the current prediction confidence.
                                float confidence = data[4];

                                // Check if the confidence is above a certain threashold.
                                if (confidence >= DNN_MINIMUM_CONFIDENCE) {
                                    // Get just the class scores from the data array. Stupid pointer manipulation
                                    float *classesScores = data + 5;
                                    Mat scores(1, classList.size(), CV_32FC1, classesScores);

                                    // Find the class id with the max score for each detection.
                                    Point classID;
                                    double maxClassScore = 0;
                                    minMaxLoc(scores, 0, &maxClassScore, 0, &classID);
                                    if (maxClassScore > DNN_MINIMUM_CLASS_SCORE) {
                                        // Add confidence and class ID to vector arrays.
                                        confidences.push_back(confidence);
                                        classIDs.push_back(classID.x);

                                        // Get box data for detection.
                                        float x = data[0];
                                        float y = data[1];
                                        float w = data[2];
                                        float h = data[3];
                                        // Calculate four corner points.
                                        int left = int((x - 0.5 * w) * widthFactor);
                                        int top = int((y - 0.5 * h) * heightFactor);
                                        int width = int(w * widthFactor);
                                        int height = int(h * heightFactor);
                                        // Add CV rect to vector array.
                                        predictionBoxes.push_back(Rect(left, top, width, height));
                                    }
                                }

                                // Completely wrap offset data array by the total length of one row.
                                data += classList.size() + 5;
                            }

                            // Remove duplicate detections/average them out.
                            vector<int> NMSResults;
                            vector<Detection> finalDetections;
                            cv::dnn::NMSBoxes(predictionBoxes, confidences, DNN_MINIMUM_CLASS_SCORE, DNN_NMS_THRESH, NMSResults);
                            for (int i = 0; i < NMSResults.size(); i++) {
                                int idx = NMSResults[i];
                                Detection result;
                                result.classID = classIDs[idx];
                                result.confidence = confidences[idx];
                                result.box = predictionBoxes[idx];
                                finalDetections.push_back(result);
                            }

                            // Loop through the detections and draw overlay onto final image.
                            for (Detection detection : finalDetections)
                            {
                                // Get detection info.
                                int classID = detection.classID;
                                float confidence = detection.confidence;
                                Rect detectionBox = detection.box;
                                Scalar color = DETECTION_COLORS[classID % DETECTION_COLORS.size()];

                                // Draw detection
                                rectangle(finalImg, detectionBox, color, 3);
                                rectangle(finalImg, Point(detectionBox.x, detectionBox.y - 20), Point(detectionBox.x + detectionBox.width, detectionBox.y), color, FILLED);
                                putText(finalImg, classList[classID].c_str(), Point(detectionBox.x, detectionBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                            }

                            break;
                        }

                        /****************************************************
                        *			Track box tape targets
                        *****************************************************/
                        case TAPE_TRACKING:
                        { 
                            // Convert image from RGB to HSV.
                            cvtColor(frame, HSVImg, COLOR_BGR2HSV);
                            // Blur the image.
                            blur(HSVImg, blurImg, Size(GREEN_BLUR_RADIUS, GREEN_BLUR_RADIUS));

                            // Loop through the scalar ranges in array and detect the colored tape for each one.
                            map<string, RotatedRect> tapeObjects;
                            for (vector<Scalar> colorRange : colorRanges)
                            {
                                // Create individual HSV ranges for each tape color. (blue, yellow, green, purple, red, pink, orange)
                                inRange(blurImg, colorRange[0], colorRange[1], filterImg);                        
                                // Remove small blobs.
                                dilate(filterImg, dilateImg, KERNEL);
                                // Find countours of image.
                                findContours(dilateImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);		////RETR_TREE //// TRY CHAIN_APPROX_SIMPLE		//// Not sure what this method of detection does, but it worked before: CHAIN_APPROX_TC89_KCOS

                                // Filter out unwanted contours based on contour area.
                                vector<vector<Point>> filteredContours;
                                for (vector<Point> contour : contours)
                                {
                                    double area = contourArea(contour);
                                    if (area >= contourAreaMinLimit && area <= contourAreaMaxLimit)
                                    {
                                        filteredContours.emplace_back(contour);
                                    }
                                }

                                // Check if we have detected one or more contours.
                                if (filteredContours.size() >= 1)
                                {
                                    // Sort contours from biggest to smallest.
                                    sort(filteredContours.begin(), filteredContours.end(), [](const vector<Point>& c1, const vector<Point>& c2) { return fabs(contourArea(c1, false)) > fabs(contourArea(c2, false)); });
                                    
                                    // Find the rotated bounding rect of only the biggest contour.
                                    Mat boxPts;
                                    RotatedRect minRect = minAreaRect(filteredContours[0]);
                                    Point2f rectPoints[4];
                                    minRect.points(rectPoints);
                                    // Draw the rotated rect in the color of current color range.
                                    for (int i = 0; i < 4; i++)
                                    {
                                        line(finalImg, rectPoints[i], rectPoints[(i + 1) % 4], colorRange[2], LINE_4);
                                    }

                                    // Find the index of the currently detected tape object and then lookup and store its color.
                                    auto location = find(colorRanges.begin(), colorRanges.end(), colorRange);
                                    int index = location - colorRanges.begin();
                                    string color = colors[index];
                                    // Store the currently detected tape and its color, so we can do calculations later.
                                    tapeObjects[color] = minRect;
                                }
                            }

                            // Sort the tapeObjects based on x position from left to right. 
                            vector<pair<string, RotatedRect>> tapeObjectsSorted;
                            // Copy key-value pair from map to vector of pairs.
                            for (auto object : tapeObjects) 
                            {
                                tapeObjectsSorted.push_back(object);
                            }
                            // Sort left to right using comparator function.
                            sort(tapeObjectsSorted.begin(), tapeObjectsSorted.end(), [](const pair<string, RotatedRect>& t1, const pair<string, RotatedRect>& t2) { return t1.second.center.x < t2.second.center.x; });

                            // Grab the current frame and crop the image down to just the side of the box.
                            if (takeShapshot)
                            {
                                // Combine all of the tape objects into one large contour.
                                vector<Point2f> boundingContour;
                                for (pair<string, RotatedRect> object : tapeObjects)
                                {
                                    // Store the tape objects points
                                    Point2f points[4];
                                    object.second.points(points);

                                    // Grab all points from the RotatedRect and add them to temporary contour.
                                    for (Point2f point : points)
                                        boundingContour.push_back(point);
                                }

                                if (boundingContour.size() >= 1)
                                {
                                    // Find the convex hull of the new combined contour.
                                    Rect cropContour = boundingRect(boundingContour);
                                    // Finally, make sure the boundingContour is within frame, and then crop.
                                    Mat croppedImg = finalImg(cropContour);
                                    // Copy cropped image to finalImg.
                                    croppedImg.copyTo(finalImg);
                                }
                            }

                            // This section of code is for the future.
                            // This is for tracking the chessboard.
                            // vector<Point2f> vImagePoints;
                            // vImagePoints.emplace_back(Point2f(0.0, 0.0));
                            // int targetPositionX = 0; 
                            // int targetPositionY = 0;
                            // solvePNPValues = SolveObjectPose(vImagePoints, ref(finalImg), ref(frame), targetPositionX, targetPositionY);
                            break;
                        }
                    }
                }

                // Put FPS on image.
                FPSCount = FPSCounter->FramesPerSec();
                putText(finalImg, ("Camera FPS: " + to_string(VideoGetter.GetFPS())), Point(420, finalImg.rows - 40), FONT_HERSHEY_DUPLEX, 0.65, Scalar(200, 200, 200), 1);
                putText(finalImg, ("Algorithm FPS: " + to_string(FPSCount)), Point(420, finalImg.rows - 20), FONT_HERSHEY_DUPLEX, 0.65, Scalar(200, 200, 200), 1);

                // If tuning mode is enabled, then output contrast or brightness images.
                if (tuningMode)
                {
                    // m_pContrastImg.copyTo(finalImg);
                    dilateImg.copyTo(finalImg);
                }
            }
        }
        catch (const exception& e)
        {
            //SetIsStopping(true);
            // Print error to console and show that an error has occured on the screen.
            putText(finalImg, "Image Processing ERROR", Point(280, finalImg.rows - 440), FONT_HERSHEY_DUPLEX, 0.65, Scalar(0, 0, 250), 1);
            cout << "\nWARNING: MAT corrupt or a runtime error has occured! Frame has been dropped." << "\n" << e.what() << endl;
        }

        // If the program stops shutdown the thread.
        if (isStopping)
        {
            break;
        }
    }

    // Clean-up.
    isStopped = true;
}

/****************************************************************************
        Description:	Turn negative numbers into -1, positive numbers 
                        into 1, and returns 0 when 0.

        Arguments: 		DOUBLE

        Returns: 		INT
****************************************************************************/
int VideoProcess::SignNum(double val)
{
    return (double(0) < val) - (val < double(0));
}

/****************************************************************************
        Description:	Use the detected object points and real world reference
                        points to estimated the 3D pose of the object.

        Arguments: 		INPUT VECTOR, OUTPUT VECTOR

        Returns: 		OUTPUT VECTOR (6 values)
****************************************************************************/
vector<double> VideoProcess::SolveObjectPose(vector<Point2f> imagePoints, Mat &finalImg, Mat &frame, int targetPositionX, int targetPositionY)
{
    // Create instance variables.
    static int count = 0;
    Vec3d					eulerAngles;
    Mat						MTXR;
    Mat						MTXQ;
    Mat						rotationVectors;
    Mat						rotationMatrix;
    Mat						translationVectors;
    Mat						translationMatrix;
    Mat						TRNSP;
    Mat						gray;
    TermCriteria termCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.001);

    // Create a vector that stores 0s by default. 
    vector<double>	objectPosition;
    objectPosition.emplace_back(1);
    objectPosition.emplace_back(2);
    objectPosition.emplace_back(3);
    objectPosition.emplace_back(4);
    objectPosition.emplace_back(5);
    objectPosition.emplace_back(6);

    // This is for tracking the chessboard.
    // vector<Point3f> chessboards;
    // for (double i = 0; i < 9; i++)			//// These were switched around.
    // {
    // 	for (double j = 0; j < 6; j++)		//// These were switched around.
    // 	{
    // 		chessboards.emplace_back(Point3f(i, j, 0.0));
    // 	}
    // }

    // Catch any anomalies from SolvePNP process. (Like bad input data errors.)
    try
    {
        // Refine the corners from image points.
        // vector<Point2f> corners;
        cvtColor(finalImg, gray, COLOR_BGR2GRAY);
        // bool bFound = findChessboardCorners(gray, Size(6, 9), corners);			// These have a possibility of being switched around.
        cornerSubPix(gray, imagePoints, Size(11, 11), Size(-1, -1), termCriteria);
        // drawChessboardCorners(finalImg, Size(6, 9), corners, bFound);			// These have a possibility of being switched around.

        // Use real world reference points and image points to estimate object pose.
        bool success = solvePnP(objectPoints,					// Object reference points in 3D space.			
                                    imagePoints,					// Object points from the 2D camera image.
                                    cameraMatrix,				// Precalibrated camera matrix. (camera specific)
                                    distanceCoefficients,		// Precalibrated camera config. (camera specific)
                                    rotationVectors,				// Storage vector for rotation values.
                                    translationVectors,			// Storage vector for translation values.
                                    false,							// Use the provided rvec and tvec values as initial approximations of the rotation and translation vectors, and further optimize them? (useExtrensicGuess)
                                    SOLVEPNP_ITERATIVE				// Method used for the PNP problem.
                                );
        // bool success = solvePnPRansac(objectPoints,						// Object reference points in 3D space.			
        // 									imagePoints,						// Object points from the 2D camera image.
        // 									cameraMatrix,				// Precalibrated camera matrix. (camera specific)
        // 									distanceCoefficients,		// Precalibrated camera config. (camera specific)
        // 									rotationVectors,				// Storage vector for rotation values.
        // 									translationVectors,			// Storage vector for translation values.
        // 									false,							// Use the provided rvec and tvec values as initial approximations of the rotation and translation vectors, and further optimize them? (useExtrensicGuess)
        //  									100,							// Number of iterations. (adjust for performance?)
        //  									15.0,							// Inlier threshold value used by the RANSAC procedure. The parameter value is the maximum allowed distance between the observed and computed point projections to consider it an inlier.
        // 									0.99,							// Confidence value that the algorithm produces a useful result. 
        //  									noArray(),						// Output vector that contains indices of inliers in objectPoints and imagePoints.
        //  									SOLVEPNP_ITERATIVE				// Method used for the PNP problem.
        // 								);

        // If SolvePNP reports a success, then continue with calculations. Else, keep searching. 
        if (success)
        {
            // Convert the rotation matrix from the solvePNP function to a rotation vector, or vise versa.
            Rodrigues(rotationVectors, rotationMatrix);

            // Calculate the camera x, y, z translation.
            transpose(rotationMatrix, TRNSP);
            translationMatrix = -TRNSP * translationVectors;

            // Calculate the pitch, roll, yaw angles of the camera.
            eulerAngles = RQDecomp3x3(rotationMatrix, MTXR, MTXQ);

            // Store the calculated object values in the vector.
            objectPosition.at(0) = translationMatrix.at<double>(0);
            objectPosition.at(1) = translationMatrix.at<double>(1);
            objectPosition.at(2) = translationMatrix.at<double>(2);
            objectPosition.at(3) = eulerAngles[0];
            objectPosition.at(4) = eulerAngles[1];
            objectPosition.at(5) = eulerAngles[2];

            // Draw axis vectors.
            drawFrameAxes(finalImg, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors, 20.0);

            // Print status onto image.
            putText(finalImg, "PNP Status: found match!", Point(50, finalImg.rows - 440), FONT_HERSHEY_DUPLEX, 0.40, Scalar(0, 0, 250), 1);
        }
        else
        {
            // If the object is not found, then put 0s in the vector.
            objectPosition.emplace_back(0);
            objectPosition.emplace_back(0);
            objectPosition.emplace_back(0);
            objectPosition.emplace_back(0);
            objectPosition.emplace_back(0);
            objectPosition.emplace_back(0);

            // Print status onto image.
            putText(finalImg, "PNP Status: searching...", Point(50, finalImg.rows - 440), FONT_HERSHEY_DUPLEX, 0.40, Scalar(0, 0, 250), 1);
        }

        // Reset toggle if the code ran successfully.
        count = 0;
    }
    catch (const exception& e)
    {
        // Print status on screen.
        putText(finalImg, "PNP Status: point data unsolvable...", Point(50, finalImg.rows - 440), FONT_HERSHEY_DUPLEX, 0.40, Scalar(0, 0, 250), 1);

        // Only print the message to the console once per fail.
        if (count <= 100)
        {
            // Print message to console.
            cout << "\nMESSAGE: SolvePNP was unable to process the image data. Moving on...\n" << e.what();

            // Add one error count to toggle.
            count++;
        }
    }

    // Return useless stuff for now.
    return objectPosition;
}

/****************************************************************************
        Description:	Signals the thread to stop.

        Arguments: 		BOOL

        Returns: 		Nothing
****************************************************************************/
void VideoProcess::SetIsStopping(bool isStopping)
{
    this->isStopping = isStopping;
}

/****************************************************************************
        Description:	Gets if the thread has stopped.

        Arguments: 		None

        Returns: 		BOOL
****************************************************************************/
bool VideoProcess::GetIsStopped()
{
    return isStopped;
}

/****************************************************************************
        Description:	Gets the current FPS of the thread.

        Arguments: 		None

        Returns: 		INT
****************************************************************************/
int VideoProcess::GetFPS()
{
    return FPSCount;
}