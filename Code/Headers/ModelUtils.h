/****************************************************************************
			Description:	Defines the ModelUtils Functions

			Classes:		None

			Project:		MATE 2022

			Copyright 2021 MST Design Team - Underwater Robotics.
****************************************************************************/
#ifndef ModelUtils_h
#define ModelUtils_h

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "edgetpu.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

using namespace std;

// Define structs.
struct Detection
{
    int classID;
    float confidence;
    Rect box;
};
///////////////////////////////////////////////////////////////////////////////

/****************************************************************************
    Description:	Build a EdgeTPU Interpreter to run inference with the
                    CoralUSB Accelerator.

    Arguments: 		CONST FLATBUFFERMODEL, EDGETPUCONTEXT

    Returns: 		UNIQUE_PTR<TFLITE::INTERPRETER> interpreter
****************************************************************************/
inline unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpuContext)
{
	// Create instance variables.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<tflite::Interpreter> interpreter;

	// Create a resolver to delegate which operations are ran on the CPU or TPU.
	resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
	// Build the Tensorflow Interpreter.
	if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
    {
	    cout << "Failed to build Tensorflow interpreter." << endl;
	}

	// Bind given context with interpreter.
	interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpuContext);
	interpreter->SetNumThreads(1);
	if (interpreter->AllocateTensors() != kTfLiteOk) 
    {
	    cout << "Failed to allocate tensors." << endl;
	}

	// Return pointer to built interpreter.
	return interpreter;
}

/****************************************************************************
    Description:	Build a Tensorflow Interpreter to run inference with the
                    with the CPU.

    Arguments: 		CONST FLATBUFFERMODEL

    Returns: 		UNIQUE_PTR<TFLITE::INTERPRETER> interpreter
****************************************************************************/
inline unique_ptr<tflite::Interpreter> BuildInterpreter(const tflite::FlatBufferModel& model)
{
  	// Create instance variables.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<tflite::Interpreter> interpreter;

	// Build the Tensorflow Interpreter.
	if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
    {
		cout << "Failed to build interpreter." << endl;
	}
	interpreter->SetNumThreads(1);
	if (interpreter->AllocateTensors() != kTfLiteOk)
    {
		cout << "Failed to allocate tensors." << endl;
	}

	// Return pointer to built interpreter.
	return interpreter;
}

/****************************************************************************
    Description:	Returns input tensor shape in the form {height, width, channels}.

    Arguments: 		CONST TFLITE::INTERPRETER* interpreter, INT index

    Returns: 		ARRAY<INT, 3>
****************************************************************************/
inline array<int, 3> GetInputShape(const tflite::Interpreter& interpreter, int index = 0)
{
    // Get shape of tensors.
    const int tensorIndex = interpreter.inputs()[index];
    const TfLiteIntArray* dims = interpreter.tensor(tensorIndex)->dims;

    // Format and return shape.
    return array<int, 3>{dims->data[1], dims->data[2], dims->data[3]};
}

/****************************************************************************
    Description:	Returns output tensor shape in the form {height, width, channels}.

    Arguments: 		CONST TFLITE::INTERPRETER* interpreter, INT index

    Returns: 		ARRAY<INT, 3>
****************************************************************************/
inline array<int, 3> GetOutputShape(const tflite::Interpreter& interpreter, int index = 0)
{
    // Get shape of tensors.
    const int tensorIndex = interpreter.outputs()[index];
    const TfLiteIntArray* dims = interpreter.tensor(tensorIndex)->dims;

    // Format and return shape.
    return array<int, 3>{dims->data[1], dims->data[2], dims->data[3]};
}

/****************************************************************************
    Description:	Runs inference on the model given the interpreter for it.

        YOLOv5 predicts 25200 grid_cells when fed with a (3, 640, 640) image 
        (Three detection layers for small, medium, and large objects same size as input with same bit depth). 
        Each grid_cell is a vector composed by (5 + num_classes) values where the 5 values are [objectness_score, Xc, Yc, W, H].
        Output would be [1, 25200, 13] for a model with eight classes.

        Check out https://pub.towardsai.net/yolov5-m-implementation-from-scratch-with-pytorch-c8f84a66c98b
        for some great info.

    Arguments: 		CONST VECTOR<UINT8> input data, TFLITE::INTERPRETER* interpreter, FLOAT confidence

    Returns: 		VECTOR<FLOAT>
****************************************************************************/
inline vector<vector<Detection>> RunInference(Mat& inputImage, tflite::Interpreter* interpreter, float confidence)
{
    // Create instance variables.
    vector<vector<vector<float>>> outputData;
    vector<size_t> outputShapes;
    vector<vector<Detection>> objects;

    // Get model image size and store given image size.
    array<int, 3> inputShape = GetInputShape(*interpreter);
    int originalInputImageWidth = inputImage.cols;
    int originalInputImageHeight = inputImage.rows;

    // Resize frame to match model size.
    resize(inputImage, inputImage, Size(inputShape[0], inputShape[1]));
    // Check model size and make sure it matches the given input image.
    if (inputShape[0] == inputImage.rows && inputShape[1] == inputImage.cols)
    {
        // Create a vector input image mat into 1 dimension.
        vector<uint8_t> inputData(inputImage.begin<uint8_t>(), inputImage.end<uint8_t>());
        // Create a new tensor and copy our input data into it.
        uint8_t* input = interpreter->typed_input_tensor<uint8_t>(interpreter->inputs()[0]);
        memcpy(input, inputData.data(), inputData.size());

        // Run inference.
        TfLiteStatus status = interpreter->Invoke();
        // Check if inference ran OK.
        if (status != kTfLiteOk)
        {
            cout << "Failed to run inference!!" << endl;
        }
        else
        {
            // Get outputs.
            const auto& outputIndices = interpreter->outputs();
            const int numOutputs = outputIndices.size();
            // Set tensor output shape and resize output data to match.
            outputShapes.resize(numOutputs);
            outputData.resize(numOutputs);

            // Loop through output tensors and extract data.
            for (int i = 0; i < numOutputs; ++i) 
            {
                // Get tensor.
                const auto* outTensor = interpreter->tensor(outputIndices[i]);
                assert(outTensor != nullptr);
                outputShapes[i] = outTensor->bytes / sizeof(uint8_t);

                // Check tensor type. UINT8 == TPU, FLOAT32 == GPU
                if (outTensor->type == kTfLiteUInt8) 
                {
                    // Get data out of tensor.
                    const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
                    
                    // Get the dimensions of the output layer of the model.
                    array<int, 3> outputShape = GetOutputShape(*interpreter, i);
                    // Resize vector to match output size.
                    outputData[i].resize(outputShape[0]);                    

                    // Reshape and store data in more usable vector.
                    // Loop through 1 dimensional array of length outputshape[0] * outputshape[1] and reshape it into yolo_grid_cellsx(5 + number of classes.)
                    int totalNumValuesCounter = 0;
                    for (int j = 0; j < outputShape[0]; ++j) 
                    {
                        // Resize vector to match output size.
                        outputData[i][j].resize(outputShape[1]);

                        // Loop through the next (5 + num classes) elements.
                        for (int k = 0; k < outputShape[1]; ++k)
                        {
                            // Get actual data stored in tensor and scale it.
                            outputData[i][j][k] = (output[totalNumValuesCounter] - outTensor->params.zero_point) * outTensor->params.scale;
                            // outputData[i][j][k] = output[totalNumValuesCounter];

                            // Increment counter.
                            totalNumValuesCounter++;
                        }
                    }
                } 
                else if (outTensor->type == kTfLiteFloat32)
                {
                    // Get data out of tensor.
                    const float* output = interpreter->typed_output_tensor<float>(i);
                    
                    // Get the dimensions of the output layer of the model.
                    array<int, 3> outputShape = GetOutputShape(*interpreter, i);
                    // Resize vector to match output size.
                    outputData[i].resize(outputShape[0]);                    

                    // Reshape and store data in more usable vector.
                    // Loop through 1 dimensional array of length outputshape[0] * outputshape[1] and reshape it into yolo_grid_cellsx(5 + number of classes.)
                    int totalNumValuesCounter = 0;
                    for (int j = 0; j < outputShape[0]; ++j) 
                    {
                        // Resize vector to match output size.
                        outputData[i][j].resize(outputShape[1]);

                        // Loop through the next (5 + num classes) elements.
                        for (int k = 0; k < outputShape[1]; ++k)
                        {
                            // Get actual data stored in tensor and scale it.
                            outputData[i][j][k] = (output[totalNumValuesCounter] - outTensor->params.zero_point) * outTensor->params.scale;
                            // outputData[i][j][k] = output[totalNumValuesCounter];

                            // Increment counter.
                            totalNumValuesCounter++;
                        }
                    }
                } 
                else 
                {
                    // If Tensor is curropted or nonsensical.
                    cout << "Tensor " << outTensor->name << " has unsupported output type: " << outTensor->type << endl;
                }
            }
                
            // Set size of object vector.
            objects.resize(outputData.size());
            // Loop through detections and calulate rect coords for each, then store the calculated data into a new Object struct if score is higher than given threshold.
            // Detections have format {xmin, ymin, width, height, conf, class0, class1, ...}
            for (int i = 0; i < outputData.size(); ++i)
            {
                for (int j = 0; j < outputData[i].size(); ++j)
                {
                    // Get detection score/confidence.
                    float predConfidence = outputData[i][j][4];
                    // Check if score is greater than or equal to minimum confidence.
                    if (predConfidence >= confidence)
                    {
                        // Get the class id.
                        Point classID;
                        double maxClassScore = 0;
                        Mat scores(1, (outputData[i][j].size() - 5), CV_32FC1, &(outputData[i][j][5]));
                        minMaxLoc(scores, 0, &maxClassScore, 0, &classID);
                        int id = classID.x;
                        // Calculate bounding box location and scale to input image.
                        int xmin = outputData[i][j][0] * originalInputImageWidth;
                        int ymin = outputData[i][j][1] * originalInputImageHeight;
                        int width = outputData[i][j][2] * originalInputImageWidth;
                        int height = outputData[i][j][3] * originalInputImageHeight;
                        
                        // Create name object variable and store info inside of it.
                        Detection object;
                        object.classID = id;
                        object.box.x = xmin;
                        object.box.y = ymin;
                        object.box.width = width;
                        object.box.height = height;
                        object.confidence = predConfidence;
                        objects[i].push_back(object);
                    }
                }
            }
        }
    }
    else
    {
        // Print warning.
        cout << "WARNING: TFLITE model size and input image size don't match! Model size is " << inputShape[0] << "x" << inputShape[1] << " (" << inputShape[2] << " channels)" << endl;
    }

    // // Print useful model information.
    // cout << "--------------------------[ MODEL INFO ]-----------------------------" << endl;
    // cout << "inputs : " << interpreter->inputs().size() << "\n";
    // cout << "inputs(0) name : " << interpreter->GetInputName(0) << "\n";
    // cout << "tensors size: " << interpreter->tensors_size() << "\n";
    // cout << "nodes size: " << interpreter->nodes_size() << "\n";
    // int startinput = clock();
    // int input = interpreter->inputs()[0];
    // cout << "input.1 : " << input <<"\n";
    // const vector<int> inputs = interpreter->inputs();
    // const vector<int> outputs = interpreter->outputs();
    // cout << "number of inputs: " <<inputs.size() << "\n";
    // cout << "number of outputs: " <<outputs.size() << "\n";
    // TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    // int test0 = dims->data[0];
    // int wanted_channels = dims->data[3];
    // int wanted_height = dims->data[1];
    // int wanted_width = dims->data[2];
    // int test4 = dims->data[4];
    // int test5 = dims->data[5];
    // int test6 = dims->data[6];
    // cout << "type of input tensor: " << interpreter->tensor(input)->type << endl;
    // cout << "height, width, channels of input : " << wanted_height << " " << wanted_width << " "<< wanted_channels <<  " " << test0 << " " << test4 << " " << test5 << " " << test6 << endl;
    // cout << "---------------------------------------------------------------------" << endl;

    // Return result vector.
    return objects;
}


/****************************************************************************
    Description:	Runs inference on the onnx model.

    Arguments: 		CONST VECTOR<UINT8> input data, TFLITE::INTERPRETER* interpreter, FLOAT confidence, INT imageSize

    Returns: 		VECTOR<FLOAT>
****************************************************************************/
inline vector<vector<Detection>> RunONNXInference(const Mat &inputImage, cv::dnn::Net &onnxModel, float confidence, int modelImgSize, int classListSize)
{
    // Create instance variables.
    vector<vector<Detection>> objects;

    // Calculate the frame width, length, and max size.
    int frameWidth = inputImage.cols;
    int frameHeight = inputImage.rows;
    int maxRes = MAX(frameWidth, frameHeight);
    // Make a new square mat with the masRes size.
    Mat resized = Mat::zeros(maxRes, maxRes, CV_8UC3);
    // Copy the camera image into the new resized mat.
    inputImage.copyTo(resized(Rect(0, 0, frameWidth, frameHeight)));
    // resize(frame, frame, Size(DNN_MODEL_IMAGE_SIZE, DNN_MODEL_IMAGE_SIZE));
    
    // Resize to 640x640, normalize to [0,1] and swap red and blue channels. This creates a 4D blob from the image.
    Mat result;
    cv::dnn::blobFromImage(inputImage, result, 1.0 / 255.0, Size(modelImgSize, modelImgSize), Scalar(), true, false);
    // Set the model's current input image.
    onnxModel.setInput(result);

    // Forward image through model layers and get the resulting predictions. This is the heavy comp shit.
    vector<Mat> predictions;
    onnxModel.forward(predictions, onnxModel.getUnconnectedOutLayersNames());
    
    // Get image and model width and height ratios.
    double widthFactor = double(frameWidth) / modelImgSize;
    double heightFactor = double(frameHeight) / modelImgSize;
    // Get class and detection data from output result.
    float *data = (float*)predictions[0].data;

    // Set size of 1st dimension in object array to 1. ONNX detection will only be using 1 model.
    objects.resize(1);

    // Loop through each prediction. This array has 25,200 positions where each position is upto nclasses-length 1D array. 
    // Each 1D array holds the data of one detection. The 4 first positions of this array are the xywh coordinates 
    // of the bound box rectangle. The fifth position is the confidence level of that detection. The 6th up to 85th 
    // elements are the scores of each class. For COCO with 80 classes outputs will be shape(n,85) with 
    // 85 dimension = (x,y,w,h,object_conf, class0_conf, class1_conf, ...)
    // I'm using ++i because it actually avoids a copy every iteration.
    for (int i = 0; i < 25200; ++i) 
    {
        // Get the current prediction confidence.
        float detectionConf = data[4];

        // Check if the confidence is above a certain threashold.
        if (detectionConf >= confidence) 
        {
            // Get just the class scores from the data array. Stupid tricky pointer manipulation, look out. Cuts off x, y, width, height, conf.
            float *classesScores = data + 5;
            Mat scores(1, classListSize, CV_32FC1, classesScores);
            // Find the class id with the max score for each detection.
            Point classID;
            double maxClassScore = 0;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classID);

            // Get box data for detection.
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            // Calculate xmin, ymin, width, and height for input image.
            int left = int((x - 0.5 * w) * widthFactor);
            int top = int((y - 0.5 * h) * heightFactor);
            int width = int(w * widthFactor);
            int height = int(h * heightFactor);
            // Create detection struct to store object prediction data in.
            Detection object;
            // Add data to detection object.
            object.box.x = left;
            object.box.y = top;
            object.box.width = width;
            object.box.height = height;
            object.confidence = detectionConf;
            object.classID = classID.x;
            // Add Detection struct to objects vector.
            objects[0].emplace_back(object);
        }

        // Completely wrap offset data array by the total length of one row.
        data += classListSize + 5;
    }

    // Return detected objects.
    return objects;
}
///////////////////////////////////////////////////////////////////////////////
#endif