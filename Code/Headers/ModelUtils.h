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
    Description:	Runs inference on the model given the interpreter for it.

    Arguments: 		CONST VECTOR<UINT8> input data, TFLITE::INTERPRETER* interpreter

    Returns: 		VECTOR<FLOAT>
****************************************************************************/
inline vector<Detection> RunInference(const Mat& inputImage, tflite::Interpreter* interpreter)
{
    // Create instance variables.
    vector<vector<uint8_t>> outputData;
    vector<size_t> outputShapes;
    vector<Detection> objects;

    // Check model size and make sure it matches the given input image.
    array<int, 3> shape = GetInputShape(*interpreter);
    if (shape[0] == inputImage.rows && shape[1] == inputImage.cols)
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
                    const int numValues = outTensor->bytes / sizeof(uint8_t);
                    const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
                    const size_t outputTensorSize = outputShapes[i];
                    // Resize vector in output data to match.
                    outputData[i].resize(outputTensorSize);

                    // Store data in more usable vector.
                    for (int j = 0; j < numValues; ++j) 
                    {
                        outputData[i][j] = (output[j] - outTensor->params.zero_point) * outTensor->params.scale;
                        // outputData[i][j] = output[j];
                    }
                } 
                else if (outTensor->type == kTfLiteFloat32)
                {
                    // Get data out of tensor.
                    const int numValues = outTensor->bytes / sizeof(float);
                    const float* output = interpreter->typed_output_tensor<float>(i);
                    const size_t outputTensorSize = outputShapes[i];
                    // Resize vector in output data to match.
                    outputData[i].resize(outputTensorSize);

                    // Store data in more usable vector.
                    for (int j = 0; j < numValues; ++j) 
                    {
                        outputData[i][j] = (output[j] - outTensor->params.zero_point) * outTensor->params.scale;
                        // outputData[i][j] = output[j];
                    }
                } 
                else 
                {
                    // If Tensor is curropted or nonsensical.
                    cout << "Tensor " << outTensor->name << " has unsupported output type: " << outTensor->type << endl;
                }
            }
                
            // Loop through locations and calulate rect coords for each, then store the calculated data into a new Object struct.
            // int n = outputData[0].size();
            // cout << "LENGTH: " << n << endl;
            // for(int i = 0; i < n; i++)
            // {
            //     // Get the class id.
            //     int id = lround(outputData[1][i]);
            //     // Get detection score/confidence.
            //     float score = outputData[2][i];
            //     // Calculate bounding box location and scale to input image.
            //     int ymin = outputData[0][4 * i] * inputImage.rows;
            //     int xmin = outputData[0][4 * i + 1] * inputImage.cols;
            //     int ymax = outputData[0][4 * i + 2] * inputImage.rows;
            //     int xmax = outputData[0][4 * i + 3] * inputImage.cols;
            //     int width = xmax - xmin;
            //     int height = ymax - ymin;
                
            //     // Create name object variable and store info inside of it.
            //     Detection object;
            //     object.classID = id;
            //     object.box.x = xmin;
            //     object.box.y = ymin;
            //     object.box.width = width;
            //     object.box.height = height;
            //     object.confidence = score;
            //     // cout << "SCORE" << ymin << endl;
            //     // objects.push_back(object);
            // }
        }
    }
    else
    {
        // Print warning.
        cout << "WARNING: TFLITE model size and input image size don't match! Model size is " << shape[0] << "x" << shape[1] << " (" << shape[2] << " channels)" << endl;
    }

    // Return result vector.
    return objects;
}
///////////////////////////////////////////////////////////////////////////////
#endif