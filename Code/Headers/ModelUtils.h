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
///////////////////////////////////////////////////////////////////////////////

/****************************************************************************
    Description:	Build a EdgeTPU Interpreter to run inference with the
                    CoralUSB Accelerator.

    Arguments: 		CONST FLATBUFFERMODEL, EDGETPUCONTEXT

    Returns: 		UNIQUE_PTR<TFLITE::INTERPRETER> interpreter
****************************************************************************/
unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpuContext)
{
	// Create instance variables.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<tflite::Interpreter> interpreter;

	// Create a resolver to delegate which operations are ran on the CPU or TPU.
	resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
	// Build the Tensorflow Interpreter.
	if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
	cout << "Failed to build Tensorflow interpreter." << endl;
	}

	// Bind given context with interpreter.
	interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpuContext);
	interpreter->SetNumThreads(1);
	if (interpreter->AllocateTensors() != kTfLiteOk) {
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
unique_ptr<tflite::Interpreter> BuildInterpreter(const tflite::FlatBufferModel& model)
{
  	// Create instance variables.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<tflite::Interpreter> interpreter;

	// Build the Tensorflow Interpreter.
	if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
		std::cerr << "Failed to build interpreter." << std::endl;
	}
	interpreter->SetNumThreads(1);
	if (interpreter->AllocateTensors() != kTfLiteOk) {
		std::cerr << "Failed to allocate tensors." << std::endl;
	}

	// Return pointer to built interpreter.
	return interpreter;
}

/****************************************************************************
    Description:	Runs inference on the model given the interpreter for it.

    Arguments: 		CONST VECTOR<UINT8> input data, TFLITE::INTERPRETER* interpreter

    Returns: 		VECTOR<FLOAT>
****************************************************************************/
vector<float> RunInference(const vector<uint8_t>& inputData, tflite::Interpreter* interpreter)
{
    // Create a new tensor and copy our input data into it.
    vector<float> outputData;
    uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
    memcpy(input, inputData.data(), inputData.size());

    // Run inference.
    interpreter->Invoke();

    // Get outputs.
    const auto& outputIndices = interpreter->outputs();
    const int numOutputs = outputIndices.size();
    int outIdx = 0;

    // Loop through output tensors and extract data.
    for (int i = 0; i < numOutputs; ++i) 
    {
        // Get tensor.
        const auto* outTensor = interpreter->tensor(outputIndices[i]);
        assert(outTensor != nullptr);

        // Check tensor type. UINT8 == CPU, FLOAT32 == TPU
        if (outTensor->type == kTfLiteUInt8) 
        {
            // Get data out of tensor.
            const int numValues = outTensor->bytes;
            outputData.resize(outIdx + numValues);
            const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);

            // Store data in more usable vector.
            for (int j = 0; j < numValues; ++j) 
            {
                outputData[outIdx++] = (output[j] - outTensor->params.zero_point) * outTensor->params.scale;
            }
        } 
        else if (outTensor->type == kTfLiteFloat32)
        {
            // Get data out of tensor.
            const int numValues = outTensor->bytes / sizeof(float);
            outputData.resize(outIdx + numValues);
            const float* output = interpreter->typed_output_tensor<float>(i);

            // Store data in more usable vector.
            for (int j = 0; j < numValues; ++j)
            {
                outputData[outIdx++] = output[j];
            }
        } 
        else 
        {
            // If Tensor is curropted or nonsensical.
            cout << "Tensor " << outTensor->name << " has unsupported output type: " << outTensor->type << endl;
        }
    }

    // Return result vector.
    return outputData;
}

/****************************************************************************
    Description:	Returns input tensor shape in the form {height, width, channels}.

    Arguments: 		CONST TFLITE::INTERPRETER* interpreter, INT index

    Returns: 		ARRAY<INT, 3>
****************************************************************************/
array<int, 3> GetInputShape(const tflite::Interpreter& interpreter, int index)
{
    // Get shape of tensors.
    const int tensorIndex = interpreter.inputs()[index];
    const TfLiteIntArray* dims = interpreter.tensor(tensorIndex)->dims;

    // Format and return shape.
    return array<int, 3>{dims->data[1], dims->data[2], dims->data[3]};
}
///////////////////////////////////////////////////////////////////////////////
#endif