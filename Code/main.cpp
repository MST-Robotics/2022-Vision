#include <cstdio>
#include <chrono>
#include <string>
#include <thread>
#include <future>
#include <mutex>
#include <shared_mutex>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h>

#include "Headers/VideoGet.h"
#include "Headers/VideoProcess.h"
#include "Headers/StereoProcess.h"
#include "Headers/VideoShow.h"
#include "../Resources/rapidjson/filereadstream.h"
#include "../Resources/rapidjson/filewritestream.h"
#include "../Resources/rapidjson/writer.h"
#include "../Resources/rapidjson/document.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <networktables/NetworkTableInstance.h>
#include <vision/VisionPipeline.h>
#include <vision/VisionRunner.h>
#include <wpi/StringExtras.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>
#include <cameraserver/CameraServer.h>
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

using namespace cv;
using namespace cs;
using namespace nt;
using namespace frc;
using namespace wpi;
using namespace std;
using namespace rapidjson;

/*
	JSON format:
	{
		"team": <team number>,
		"ntmode": <"client" or "server", "client" if unspecified>
		"cameras": [
			{
				"name": <camera name>
				"path": <path, e.g. "/dev/video0">
				"pixel format": <"MJPEG", "YUYV", etc>	// optional
				"width": <video mode width>			  // optional
				"height": <video mode height>			// optional
				"fps": <video mode fps>				  // optional
				"brightness": <percentage brightness>	// optional
				"white balance": <"auto", "hold", value> // optional
				"exposure": <"auto", "hold", value>	  // optional
				"properties": [						  // optional
					{
						"name": <property name>
						"value": <property value>
					}
				],
				"stream": {							  // optional
					"properties": [
						{
							"name": <stream property name>
							"value": <stream property value>
						}
					]
				}
			}
		]
		"switched cameras": [
			{
				"name": <virtual camera name>
				"key": <network table key used for selection>
				// if NT value is a string, it's treated as a name
				// if NT value is a double, it's treated as an integer index
			}
		]
	}
 */

// Store config file.
static const char* configFile = "/boot/frc.json";
static const char* VisionTuningFilePath = "/home/pi/2022-Vision/Code/trackbar_values.json";
static const char* StereoCameraParamsPath = "/home/pi/2022-Vision/Code/stereo_params.json";
static const string YoloModelFilePath = "/home/pi/2022-Vision/YOLO_Models/COCO_v5n_Test/";

// Create namespace variables, stucts, and objects.
unsigned int team;
bool server = false;

// Create json interactable object.
Document visionTuningJSON;
Document stereoCameraParamsJSON;

// Create enums.
enum SelectionStates { TRENCH, LINE, FISH, TAPE };

// Create structs.
struct CameraConfig 
{
	string name;
	string path;
	json config;
	json streamConfig;
};

struct SwitchedCameraConfig 
{
	string name;
	string key;
};

// Declare global arrays.
vector<CameraConfig> cameraConfigs;
vector<SwitchedCameraConfig> switchedCameraConfigs;
vector<UsbCamera> cameras;
vector<CvSink> cameraSinks;
vector<CvSource> cameraSources;

raw_ostream& ParseError() 
{
	return errs() << "config error in '" << configFile << "': ";
}

/****************************************************************************
		Description:	Read camera config file from the web dashboard.

		Arguments: 		CONST JSON&

		Returns: 		BOOL
****************************************************************************/
bool ReadCameraConfig(const json& config) 
{
	// Create instance variables.
	CameraConfig camConfig;

	// Get camera name.
	try 
	{
		camConfig.name = config.at("name").get<string>();
	} 
	catch (const json::exception& e) 
	{
		ParseError() << "Could not read camera name: " << e.what() << "\n";
		return false;
	}

	// Get camera path.
	try 
	{
		camConfig.path = config.at("path").get<string>();
	} 
	catch (const json::exception& e) 
	{
		ParseError() << "Camera '" << camConfig.name << "': could not read path: " << e.what() << "\n";
		return false;
	}

	// Get stream properties.
	if (config.count("stream") != 0)
	{
		camConfig.streamConfig = config.at("stream");
	}

	camConfig.config = config;

	cameraConfigs.emplace_back(move(camConfig));
	return true;
}

/****************************************************************************
		Description:	Read config file from the web dashboard.

		Arguments: 		None

		Returns: 		BOOL
****************************************************************************/
bool ReadConfig() 
{
	// Open config file.
	error_code errorCode;
	raw_fd_istream is(configFile, errorCode);
	if (errorCode) 
	{
		errs() << "Could not open '" << configFile << "': " << errorCode.message() << "\n";
		return false;
	}

	// Parse file.
	json parseFile;
	try 
	{
		parseFile = json::parse(is);
	} 
	catch (const json::parse_error& e) 
	{
		ParseError() << "Byte: " << e.what() << "\n";
		return false;
	}

	// Check if the top level is an object.
	if (!parseFile.is_object()) 
	{
		ParseError() << "Must be JSON object!" << "\n";
		return false;
	}

	// Get team number.
	try 
	{
		team = parseFile.at("team").get<unsigned int>();
	} 
	catch (const json::exception& e) 
	{
		ParseError() << "Could not read team number: " << e.what() << "\n";
		return false;
	}

	// Get NetworkTable mode.
	if (parseFile.count("ntmode") != 0) 
	{
		try 
		{
			auto str = parseFile.at("ntmode").get<string>();
			if (equals_lower(str, "client")) 
			{
				server = false;
			} 
			else 
			{
				if (equals_lower(str, "server")) 
				{
					server = true;
				}
				else 
				{
					ParseError() << "Could not understand ntmode value '" << str << "'" << "\n";
				}
			} 
		} 
		catch (const json::exception& e) 
		{
			ParseError() << "Could not read ntmode: " << e.what() << "\n";
		}
	}

	// Read camera configs and get cameras.
	try 
	{
		for (auto&& camera : parseFile.at("cameras")) 
		{
			if (!ReadCameraConfig(camera))
			{
				return false;
			}
		}
	} 
	catch (const json::exception& e) 
	{
		ParseError() << "Could not read cameras: " << e.what() << "\n";
		return false;
	}

	return true;
}

/****************************************************************************
		Description:	Starts cameras and camera streams.

		Arguments: 		CONST CAMERACONFIG&

		Returns: 		Nothing
****************************************************************************/
void StartCamera(const CameraConfig& config) 
{
	// Print debug
	cout << "Starting camera '" << config.name << "' on " << config.path << "\n";

	// Create new CameraServer instance and start camera.
	// CameraServer* instance = CameraServer::GetInstance();
	UsbCamera camera{config.name, config.path};
	MjpegServer server = CameraServer::StartAutomaticCapture(camera);

	// Set camera parameters.
	camera.SetConfigJson(config.config);
	camera.SetConnectionStrategy(VideoSource::kConnectionKeepOpen);

	// Check for unexpected parameters.
	if (config.streamConfig.is_object())
	{
		server.SetConfigJson(config.streamConfig); 
	}

	// Store the camera video in a vector. (so we can access it later)
	CvSink cvSink = CameraServer::GetVideo(config.name);
	CvSource cvSource = CameraServer::PutVideo(config.name + "Processed", 640, 480);
	cameras.emplace_back(camera);
	cameraSinks.emplace_back(cvSink);
	cameraSources.emplace_back(cvSource);
}

/****************************************************************************
		Description:	Gets values from json file and updates networktables
						with the corresponding values.

		Arguments: 		AUTO &NetworkTable, int selectionState

		Returns: 		Nothing
****************************************************************************/
void GetJSONValues(auto &NetworkTable, int selectionState = -1)
{
	// Create instance variables.
	string state = "";

	// Convert selection state enum number into string.
	switch (selectionState)
	{
		case TRENCH:
			state = "TRENCH";
			break;
		case LINE:
			state = "LINE";
			break;
		case FISH:
			state = "FISH";
			break;
		case TAPE:
			state = "TAPE";
			break;
		default:
			state = "STEREO";
			break;
	}

	// Convert string parameter to char array.
	char* jsonState = &*state.begin();
	// Get corresponding object from JSON file based on tracking state.
	const rapidjson::Value& object = visionTuningJSON[jsonState];

	// Check if we are putting normal trackbar values or stereo values.
	if (state != "STEREO")
	{
		// Get trackbar values from json object.
		int contourAreaMinLimit = object["ContourAreaMinLimit"].GetInt();
		int contourAreaMaxLimit = object["ContourAreaMaxLimit"].GetInt();
		int hmn = object["HMN"].GetInt();
		int hmx = object["HMX"].GetInt();
		int smn = object["SMN"].GetInt();
		int smx = object["SMX"].GetInt();
		int vmn = object["VMN"].GetInt();
		int vmx = object["VMX"].GetInt();

		// Update network tables with the values from the JSON document object.
		NetworkTable->PutNumber("Contour Area Min Limit", contourAreaMinLimit);
		NetworkTable->PutNumber("Contour Area Max Limit", contourAreaMaxLimit);
		NetworkTable->PutNumber("HMN", hmn);
		NetworkTable->PutNumber("HMX", hmx);
		NetworkTable->PutNumber("SMN", smn);
		NetworkTable->PutNumber("SMX", smx);
		NetworkTable->PutNumber("VMN", vmn);
		NetworkTable->PutNumber("VMX", vmx);
	}
	else
	{
		// Get trackbar values from json object.
		int numDisparities = object["NumDisparities"].GetInt();
		int minDisparities = object["MinDisparities"].GetInt();
		int blockSize = object["BlockSize"].GetInt();
		int preFilterType = object["PreFilterType"].GetInt();
		int preFilterSize = object["PreFilterSize"].GetInt();
		int preFilterCap = object["PreFilterCap"].GetInt();
		int textureThresh = object["TextureThresh"].GetInt();
		int uniquenessRatio = object["UniquenessRatio"].GetInt();
		int speckleRange = object["SpeckleRange"].GetInt();
		int speckleWindowSize = object["SpeckleWindowSize"].GetInt();
		int disp12MaxDiff = object["Disp12MaxDiff"].GetInt();

		// Update network tables with the values from the JSON document object.
		NetworkTable->PutNumber("Stereo Num Disparities", numDisparities);
		NetworkTable->PutNumber("Stereo Min Disparity", minDisparities);
		NetworkTable->PutNumber("Stereo Block Size", blockSize);
		NetworkTable->PutNumber("Stereo PreFilter Type", preFilterType);
		NetworkTable->PutNumber("Stereo PreFilter Size", preFilterSize);
		NetworkTable->PutNumber("Stereo PreFilter Cap", preFilterCap);
		NetworkTable->PutNumber("Stereo TextureThresh", textureThresh);
		NetworkTable->PutNumber("Stereo Uniqueness Ratio", uniquenessRatio);
		NetworkTable->PutNumber("Stereo Speckle Range", speckleRange);
		NetworkTable->PutNumber("Stereo Speckle WindowSize", speckleWindowSize);
		NetworkTable->PutNumber("Stereo Disp12MaxDiff", disp12MaxDiff);
	}
}

/****************************************************************************
		Description:	Gets values from networktables and update the JSON file
						with the corresponding values.

		Arguments: 		AUTO &NetworkTable, int selectionState

		Returns: 		Nothing
****************************************************************************/
void PutJSONValues(auto &NetworkTable, int selectionState = -1)
{
	// Create instance variables.
	string state = "";

	// Convert selection state enum number into string.
	switch (selectionState)
	{
		case TRENCH:
			state = "TRENCH";
			break;
		case LINE:
			state = "LINE";
			break;
		case FISH:
			state = "FISH";
			break;
		case TAPE:
			state = "TAPE";
			break;
		default:
			state = "STEREO";
			break;
	}

	// Convert string parameter to char array.
	char* jsonState = &*state.begin();
	// Get corresponding object from JSON file based on tracking state.
	rapidjson::Value& object = visionTuningJSON[jsonState];

	// Check if we are putting normal trackbar values or stereo values.
	if (state != "STEREO")
	{
		// Get trackbar values from NetworkTables.
		int contourAreaMinLimit = NetworkTable->GetNumber("Contour Area Min Limit", 0);
		int contourAreaMaxLimit = NetworkTable->GetNumber("Contour Area Max Limit", 0);
		int hmn = NetworkTable->GetNumber("HMN", 0);
		int hmx = NetworkTable->GetNumber("HMX", 0);
		int smn = NetworkTable->GetNumber("SMN", 0);
		int smx = NetworkTable->GetNumber("SMX", 0);
		int vmn = NetworkTable->GetNumber("VMN", 0);
		int vmx = NetworkTable->GetNumber("VMX", 0);

		// Update json object with current trackbar values.
		object["ContourAreaMinLimit"].SetInt(contourAreaMinLimit);
		object["ContourAreaMaxLimit"].SetInt(contourAreaMaxLimit);
		object["HMN"].SetInt(hmn);
		object["HMX"].SetInt(hmx);
		object["SMN"].SetInt(smn);
		object["SMX"].SetInt(smx);
		object["VMN"].SetInt(vmn);
		object["VMX"].SetInt(vmx);
	}
	else
	{
		// Get trackbar values from NetworkTables.
		int numDisparities = NetworkTable->GetNumber("Stereo Num Disparities", 18);
		int minDisparities = NetworkTable->GetNumber("Stereo Min Disparity", 25);
		int blockSize = NetworkTable->GetNumber("Stereo Block Size", 50);
		int preFilterType = NetworkTable->GetNumber("Stereo PreFilter Type", 1);
		int preFilterSize= NetworkTable->GetNumber("Stereo PreFilter Size", 25);
		int preFilterCap = NetworkTable->GetNumber("Stereo PreFilter Cap", 62);
		int textureThresh = NetworkTable->GetNumber("Stereo TextureThresh", 100);
		int uniquenessRatio = NetworkTable->GetNumber("Stereo Uniqueness Ratio", 100);
		int speckleRange = NetworkTable->GetNumber("Stereo Speckle Range", 100);
		int speckleWindowSize = NetworkTable->GetNumber("Stereo Speckle WindowSize", 25);
		int disp12MaxDiff = NetworkTable->GetNumber("Stereo Disp12MaxDiff", 25);

		// Update json object with current trackbar values.
		object["NumDisparities"].SetInt(numDisparities);
		object["MinDisparities"].SetInt(minDisparities);
		object["BlockSize"].SetInt(blockSize);
		object["PreFilterType"].SetInt(preFilterType);
		object["PreFilterSize"].SetInt(preFilterSize);
		object["PreFilterCap"].SetInt(preFilterCap);
		object["TextureThresh"].SetInt(textureThresh);
		object["UniquenessRatio"].SetInt(uniquenessRatio);
		object["SpeckleRange"].SetInt(speckleRange);
		object["SpeckleWindowSize"].SetInt(speckleWindowSize);
		object["Disp12MaxDiff"].SetInt(disp12MaxDiff);
	}
}


/****************************************************************************
    Description:	Main method

    Arguments: 		None

    Returns: 		Nothing
****************************************************************************/
int main(int argc, char* argv[]) 
{
	/************************************************************************** 
	  			Read Configurations
	 * ************************************************************************/
	// Set web dashboard config path if given as argument.
	if (argc >= 2) 
	{
		configFile = argv[1];
	}

	// Read dashboard config.
	if (!ReadConfig())
	{
		return EXIT_FAILURE;
	}

	// Open vision trackbar json for reading and writing.
	FILE* jsonVisionFile = fopen(VisionTuningFilePath, "r");
	// Check if file was successfully opened.
	if (jsonVisionFile == nullptr)
	{
		cout << "ERROR: Unable to find, open, or load trackbar JSON file. Check that it exist at this path (" << VisionTuningFilePath << ") and that it is not corrupt." << endl;
		return EXIT_FAILURE;
	}
	// Attempt to open JSON file containing vision tuning params.
	// Create empty data buffer.
	char readBufferVision[65536];
	// Store opened file in buffer.
	FileReadStream readVisionFileStream(jsonVisionFile, readBufferVision, sizeof(readBufferVision));
	// Parse stream buffer into rapidjson document.
	visionTuningJSON.ParseStream(readVisionFileStream);

	// Open vision trackbar json for reading and writing.
	FILE* jsonStereoFile = fopen(StereoCameraParamsPath, "r");
	// Check if file was successfully opened.
	if (jsonStereoFile == nullptr)
	{
		cout << "Warning: Unable to find, open, or load stereo camera calibration JSON file. Check that it exist at this path (" << StereoCameraParamsPath << ") and that it is not corrupt." << endl;
	}
	else
	{
		// Attempt to open JSON file containing stereo camera calibration.
		// Create empty data buffer.
		char readBufferStereo[65536];
		// Store opened file in buffer.
		FileReadStream readStereoFileStream(jsonStereoFile, readBufferStereo, sizeof(readBufferStereo));
		// Parse stream buffer into rapidjson document.
		visionTuningJSON.ParseStream(readStereoFileStream);

		// Pull data from the JSON file to pass to the StereoProcess thread.
	}

	/**************************************************************************
	  			Start NetworkTables
	 * ************************************************************************/
	// Create instance.
	auto NetworkTablesInstance = NetworkTableInstance::GetDefault();
	auto NetworkTable = NetworkTablesInstance.GetTable("SmartDashboard");

	// Start Networktables as a client or server.
	if (server) 
	{
		cout << "Setting up NetworkTables server" << "\n";
		NetworkTablesInstance.StartServer();
	} 
	else 
	{
		cout << "Setting up NetworkTables client for team " << team << "\n";
		NetworkTablesInstance.StartClientTeam(team);
	}

	// Give network tables some time to connect before initializing values.
	this_thread::sleep_for(std::chrono::milliseconds(500));
	// Populate NetworkTables.
	NetworkTable->PutBoolean("Write JSON", false);
	NetworkTable->PutBoolean("Restart Program", false);
	NetworkTable->PutBoolean("Camera Source", false);
	NetworkTable->PutBoolean("Tuning Mode", false);
	NetworkTable->PutBoolean("Driving Mode", false);
	NetworkTable->PutBoolean("Trench Tracking Mode", false);
	NetworkTable->PutBoolean("Line Tracking Mode", false);
	NetworkTable->PutBoolean("Fish Tracking Mode", false);
	NetworkTable->PutBoolean("Tape Tracking Mode", false);
	NetworkTable->PutBoolean("Take Shapshot", false);
	NetworkTable->PutBoolean("Enable SolvePNP", false);
	NetworkTable->PutBoolean("Enable StereoVision", false);
	NetworkTable->PutBoolean("Force ONNX Model", false);
	NetworkTable->PutNumber("X Setpoint Offset", 0);
	NetworkTable->PutNumber("Contour Area Min Limit", 1211);
	NetworkTable->PutNumber("Contour Area Max Limit", 2000);
	NetworkTable->PutNumber("Center Line Tolerance", 50);
	NetworkTable->PutNumber("Neural Net Min Confidence", 0.4);
	NetworkTable->PutNumber("HMN", 48);
	NetworkTable->PutNumber("HMX", 104);
	NetworkTable->PutNumber("SMN", 0);
	NetworkTable->PutNumber("SMX", 128);
	NetworkTable->PutNumber("VMN", 0);
	NetworkTable->PutNumber("VMX", 0);
	NetworkTable->PutNumberArray("Tracking Results", vector<double> {});
	// Populate NetworkTables with adjustable stereo parameters.
    NetworkTable->PutNumber("Stereo Num Disparities", 18);
	NetworkTable->PutNumber("Stereo Min Disparity", 25);
    NetworkTable->PutNumber("Stereo Block Size", 50);
    NetworkTable->PutNumber("Stereo PreFilter Type", 1);
	NetworkTable->PutNumber("Stereo PreFilter Size", 25);
	NetworkTable->PutNumber("Stereo PreFilter Cap", 62);
	NetworkTable->PutNumber("Stereo TextureThresh", 100);
	NetworkTable->PutNumber("Stereo Uniqueness Ratio", 100);
	NetworkTable->PutNumber("Stereo Speckle Range", 100);
	NetworkTable->PutNumber("Stereo Speckle WindowSize", 25);
	NetworkTable->PutNumber("Stereo Disp12MaxDiff", 25);

	/**************************************************************************
	 			Start Cameras
	**************************************************************************/
	for (const auto& config : cameraConfigs)
	{
		StartCamera(config);
	}

	/**************************************************************************
				Load Neural Network Model for opencv CPU processing.
	**************************************************************************/
	// Create DNN model object.
	cv::dnn::Net onnxModel;
	vector<string> classList;
	// Start loading yolo model.
	try
	{
		cout << "\nAttempting to load ONNX DNN model..." << endl;
		onnxModel = cv::dnn::readNet(string(YoloModelFilePath + "best.onnx"));
		cout << "ONNX DNN Model is loaded." << endl;
		// Get class list.
		ifstream ifs(string(YoloModelFilePath + "classes.txt"));
		string line;
		while (getline(ifs, line))
		{
			classList.push_back(line);
		}
		cout << "ONNX DNN class list loaded successfully." << endl;
	}
	catch (const exception& e)
	{
		// Print error to console and show that an error has occured on the screen.
		cout << "\nWARNING: Unable to load DNN model, neural network inferencing will not work on the CPU with OpenCV." << "\n" << e.what() << endl;
	}

	/**************************************************************************
				Load Neural Network Model for EdgeTPU ASIC processing.
	**************************************************************************/
	// Create model objects.
	unique_ptr<tflite::FlatBufferModel> tfliteModel;
	shared_ptr<edgetpu::EdgeTpuContext> edgetpuContext;
	unique_ptr<tflite::Interpreter> tfliteModelInterpreter;
	// Start loading yolo model.
	try
	{
		// Attempt to open the tflite model.
		cout << "Attempting to open EdgeTPU TFLITE model..." << endl;
		tfliteModel = tflite::FlatBufferModel::BuildFromFile(string(YoloModelFilePath + "best.tflite").c_str());
		cout << "EdgeTPU TFLITE Model is opened." << endl;
		// Attempt to open CoralUSB Accelerator.
		cout << "Attempting to open CoralUSB device..." << endl;
		edgetpuContext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
		// Check if device was opened properly.
		if (!edgetpuContext)
		{
			cout << "Unable to open CoralUSB device. Defualting to CPU Tensorflow interpreter." << endl;
			tfliteModelInterpreter = BuildInterpreter(*tfliteModel);
			cout << "WARNING: CPU Tensorflow interpreter has been loaded and is probably broken." << endl;
		}
		else
		{
			cout << "CoralUSB device has been opened..." << endl;
			cout << "Attempting to load model onto device and create interpreter..." << endl;
			// Create a Tensorflow interpreter with the opened model and the opened EdgeTPU device.
			tfliteModelInterpreter = BuildEdgeTpuInterpreter(*tfliteModel, edgetpuContext.get());
			cout << "SUCCESS: EdgeTPU Tensorflow interpreter has been loaded with CoralUSB device." << endl;
		}		
	}
	catch (const exception& e)
	{
		// Print error to console and show that an error has occured on the screen.
		cout << "\nWARNING: Unable to either open TFLITE model or load open EdgeTPU ASIC, neural network inferencing will not work for the CoralUSB Accelerator." << "\n" << e.what() << endl;
	}

	/**************************************************************************
	 			Start Image Processing on Camera 0
	**************************************************************************/
	if (cameraSinks.size() >= 1) 
	{
		// Create object pointers for threads.
		VideoGet VideoGetter;
		VideoProcess VideoProcessor;
		StereoProcess StereoProcessor;
		VideoShow VideoShower;

		// Preallocate image objects.
		Mat	visionFrame(480, 640, CV_8U, 1);
		Mat leftStereoFrame(480, 640, CV_8U, 1);
		Mat rightStereoFrame(480, 640, CV_8U, 1);
		Mat stereoImg(480, 640, CV_8U, 1);
		Mat finalImg(480, 640, CV_8U, 1);

		// Create a global instance of mutex to protect it.
		shared_timed_mutex VisionMutexGet;
		shared_timed_mutex VisionMutexShow;
		shared_timed_mutex StereoMutexGet;
		shared_timed_mutex StereoMutexShow;

		// Load values for stereo trackbars from JSON and set them in NetworkTables.
		GetJSONValues(NetworkTable);

		// Vision options and values.
		int targetCenterX = 0;
		int targetCenterY = 0;
		int centerLineTolerance = 0;
		double contourAreaMinLimit = 0;
		double contourAreaMaxLimit = 0;
		float neuralNetworkMinConfidence = 0.4;
		bool writeJSON = false;
		bool stopProgam = false;
		bool cameraSourceIndex = false;
		bool tuningMode = false;
		bool drivingMode = false;
		bool takeShapshot = false;
		bool enableSolvePNP = false;
		bool enableStereoVision = false;
		bool forceONNXModel = false;
		bool valsSet = false;
		int trackingMode = VideoProcess::LINE_TRACKING;
		int selectionState = LINE;
		vector<int> trackbarValues {1, 255, 1, 255, 1, 255};
		vector<double> trackingResults {};
		vector<double> solvePNPValues {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

		// Start classes multi-threading.
		thread VideoGetThread(&VideoGet::StartCapture, &VideoGetter, ref(visionFrame), ref(leftStereoFrame), ref(rightStereoFrame), ref(cameraSourceIndex), ref(drivingMode), ref(cameraSinks), ref(VisionMutexGet), ref(StereoMutexGet));
		thread VideoProcessThread(&VideoProcess::Process, &VideoProcessor, ref(visionFrame), ref(finalImg), ref(targetCenterX), ref(targetCenterY), ref(centerLineTolerance), ref(contourAreaMinLimit), ref(contourAreaMaxLimit), ref(tuningMode), ref(drivingMode), ref(trackingMode), ref(takeShapshot), ref(enableSolvePNP), ref(trackbarValues), ref(trackingResults), ref(solvePNPValues), ref(classList), ref(onnxModel), ref(tfliteModelInterpreter), ref(neuralNetworkMinConfidence), ref(forceONNXModel), ref(VideoGetter), ref(VisionMutexGet), ref(VisionMutexShow));
		thread StereoProcessThread(&StereoProcess::Process, &StereoProcessor, ref(leftStereoFrame), ref(rightStereoFrame), ref(stereoImg), ref(enableStereoVision), ref(VideoGetter), ref(StereoMutexGet), ref(StereoMutexShow));
		thread VideoShowerThread(&VideoShow::ShowFrame, &VideoShower, ref(finalImg), ref(stereoImg), ref(cameraSources), ref(VisionMutexShow), ref(StereoMutexShow));
		
		while (1)
		{
			try
			{
				// Check if any of the threads have stopped.
				if (!VideoGetter.GetIsStopped() && !VideoProcessor.GetIsStopped() && !StereoProcessor.GetIsStopped() && !VideoShower.GetIsStopped() && !stopProgam)
				{
					// Get NetworkTables data.
					writeJSON = NetworkTable->GetBoolean("Write JSON", false);
					stopProgam = NetworkTable->GetBoolean("Restart Program", false);
					cameraSourceIndex = NetworkTable->GetBoolean("Camera Source", false);
					tuningMode = NetworkTable->GetBoolean("Tuning Mode", false);
					drivingMode = NetworkTable->GetBoolean("Driving Mode", false);
					bool trenchMode = NetworkTable->GetBoolean("Trench Tracking Mode", false);
					bool lineMode = NetworkTable->GetBoolean("Line Tracking Mode", false);
					bool fishMode = NetworkTable->GetBoolean("Fish Tracking Mode", true);
					bool tapeMode = NetworkTable->GetBoolean("Tape Tracking Mode", false);
					takeShapshot = NetworkTable->GetBoolean("Take Shapshot", false);
					enableSolvePNP = NetworkTable->GetBoolean("Enable SolvePNP", false);
					enableStereoVision = NetworkTable->GetBoolean("Enable StereoVision", false);
					forceONNXModel = NetworkTable->GetBoolean("Force ONNX Model", false);
					centerLineTolerance = NetworkTable->GetNumber("Center Line Tolerance", 50);
					contourAreaMinLimit = NetworkTable->GetNumber("Contour Area Min Limit", 1211.0);
					contourAreaMaxLimit = NetworkTable->GetNumber("Contour Area Max Limit", 2000);
					neuralNetworkMinConfidence = NetworkTable->GetNumber("Neural Net Min Confidence", 0.4);
					trackbarValues[0] = int(NetworkTable->GetNumber("HMN", 1));
					trackbarValues[1] = int(NetworkTable->GetNumber("HMX", 255));
					trackbarValues[2] = int(NetworkTable->GetNumber("SMN", 1));
					trackbarValues[3] = int(NetworkTable->GetNumber("SMX", 255));
					trackbarValues[4] = int(NetworkTable->GetNumber("VMN", 1));
					trackbarValues[5] = int(NetworkTable->GetNumber("VMX", 255));
					// Tracking mode selection state logic.
					switch (selectionState)
					{
						case TRENCH:
							// If line mode is selected move to other state.
							if (lineMode)
							{
								// Deselect trench tracking mode.
								NetworkTable->PutBoolean("Trench Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::LINE_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = LINE;
							}
							// If fish mode is selected move to other state.
							else if (fishMode)
							{
								// Deselect trench tracking mode.
								NetworkTable->PutBoolean("Trench Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::FISH_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = FISH;
							}
							// If tape mode is selected move to other state.
							else if (tapeMode)
							{
								// Deselect trench tracking mode.
								NetworkTable->PutBoolean("Trench Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::TAPE_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = TAPE;
							}
							else
							{
								// Make sure trench mode is true while in this state.
								NetworkTable->PutBoolean("Trench Tracking Mode", true);
								// Only set mode specific values once.
								if (!valsSet)
								{
									// Update networktables values.
									GetJSONValues(NetworkTable, selectionState);
									// Update setVals flag.
									valsSet = true;
								}
							}
							break;

						case LINE:
							// If trench mode is selected move to other state.
							if (trenchMode)
							{
								// Deselect line tracking mode.
								NetworkTable->PutBoolean("Line Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::TRENCH_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = TRENCH;
							}
							// If fish mode is selected move to other state.
							else if (fishMode)
							{
								// Deselect line tracking mode.
								NetworkTable->PutBoolean("Line Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::FISH_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = FISH;
							}
							// If tape mode is selected move to other state.
							else if (tapeMode)
							{
								// Deselect line tracking mode.
								NetworkTable->PutBoolean("Line Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::TAPE_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = TAPE;
							}
							else
							{
								// Make sure trench mode is true while in this state.
								NetworkTable->PutBoolean("Line Tracking Mode", true);
								// Only set mode specific values once.
								if (!valsSet)
								{
									// Update networktables values.
									GetJSONValues(NetworkTable, selectionState);
									// Update setVals flag.
									valsSet = true;
								}
							}
							break;

						case FISH:
							// If trench mode is selected move to other state.
							if (trenchMode)
							{
								// Deselect line tracking mode.
								NetworkTable->PutBoolean("Fish Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::TRENCH_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = TRENCH;
							}
							// If line mode is selected move to other state.
							else if (lineMode)
							{
								// Deselect trench tracking mode.
								NetworkTable->PutBoolean("Fish Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::LINE_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = LINE;
							}
							// If tape mode is selected move to other state.
							else if (tapeMode)
							{
								// Deselect line tracking mode.
								NetworkTable->PutBoolean("Fish Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::TAPE_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = TAPE;
							}
							else
							{
								// Make sure trench mode is true while in this state.
								NetworkTable->PutBoolean("Fish Tracking Mode", true);
								// Only set mode specific values once.
								if (!valsSet)
								{
									// Update networktables values.
									GetJSONValues(NetworkTable, selectionState);
									// Update setVals flag.
									valsSet = true;
								}
							}
							break;

						case TAPE:
							// If trench mode is selected move to other state.
							if (trenchMode)
							{
								// Deselect tape tracking mode.
								NetworkTable->PutBoolean("Tape Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::TRENCH_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = TRENCH;
							}
							// If line mode is selected move to other state.
							else if (lineMode)
							{
								// Deselect trench tracking mode.
								NetworkTable->PutBoolean("Tape Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::LINE_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = LINE;
							}
							// If fish mode is selected move to other state.
							else if (fishMode)
							{
								// Deselect line tracking mode.
								NetworkTable->PutBoolean("Tape Tracking Mode", false);
								// Set tracking mode.
								trackingMode = VideoProcess::FISH_TRACKING;
								// Set update values toggle.
								valsSet = false;
								// Store current tackbar values for this tracking state into memory JSON.
								PutJSONValues(NetworkTable, selectionState);
								// Move to other state.
								selectionState = FISH;
							}
							else
							{
								// Make sure tape mode is true while in this state.
								NetworkTable->PutBoolean("Tape Tracking Mode", true);
								// Only set mode specific values once.
								if (!valsSet)
								{
									// Update networktables values.
									GetJSONValues(NetworkTable, selectionState);
									// Update setVals flag.
									valsSet = true;
								}
							}
							break;
					}

					// Put NetworkTables data.
					NetworkTable->PutNumber("Target Center X", (targetCenterX + int(NetworkTable->GetNumber("X Setpoint Offset", 0))));
					NetworkTable->PutNumber("Target Width", targetCenterY);
					if (!trackingResults.empty())
					{
						NetworkTable->PutBoolean("Line Is Vertical", trackingResults[0]);
						NetworkTable->PutNumberArray("Tracking Results", trackingResults);
					}
					// NetworkTable->PutNumber("SPNP X Dist", solvePNPValues[0]);
					// NetworkTable->PutNumber("SPNP Y Dist", solvePNPValues[1]);
					// NetworkTable->PutNumber("SPNP Z Dist", solvePNPValues[2]);
					// NetworkTable->PutNumber("SPNP Roll", solvePNPValues[3]);
					// NetworkTable->PutNumber("SPNP Pitch", solvePNPValues[4]);
					// NetworkTable->PutNumber("SPNP Yaw", solvePNPValues[5]);

					// Write current memory JSON document to disk if button is selected.
					if (writeJSON)
					{ 
						// Make sure to store the current trackbar values in current state.
						PutJSONValues(NetworkTable, selectionState);
						// Always store the stereo values.
						PutJSONValues(NetworkTable);

						// Create string buffer for storing the current json object values.
						StringBuffer buffer;
						Writer<StringBuffer> writer(buffer);
						visionTuningJSON.Accept(writer);
						
						// Convert the buffer to a Cstring.
						const char* output = buffer.GetString();
						cout << "JSON Data: " << output << endl;

						// Reopen and clear json file.
						FILE* file = fopen(VisionTuningFilePath, "w");
						// Write string contents to file and close it.
						fwrite(output , sizeof(output[0]), strlen(output), file);
						fclose(file);

						// Unselect toggle button after writing is done.
						NetworkTable->PutBoolean("Write JSON", false);
					}
					
					// Sleep.
					this_thread::sleep_for(std::chrono::milliseconds(20));

					// Print debug info.
					//cout << "Getter FPS: " << VideoGetter.GetFPS(1) << "\n";
					//cout << "Processor FPS: " << VideoProcessor.GetFPS() << "\n";
					//cout << "Shower FPS: " << VideoShower.GetFPS() << "\n";
				}
				else
				{
					// Notify other threads the program is stopping.
					VideoGetter.SetIsStopping(true);
					VideoProcessor.SetIsStopping(true);
					StereoProcessor.SetIsStopping(true);
					VideoShower.SetIsStopping(true);
					break;
				}
			}
			catch (const exception& e)
			{
				cout << "CRITICAL: A main thread error has occured!" << "\n";
			}
		}

		// Stop all threads.
		VideoGetThread.join();
		VideoProcessThread.join();
		StereoProcessThread.join();
		VideoShowerThread.join();

		// Close opened file stream.
		fcloseall();

		// Print that program has safely and successfully shutdown.
		cout << "All threads have been released! Program will now stop..." << "\n";

		// Reset restart program button to avoid an endless restart loop.
		NetworkTable->PutBoolean("Restart Program", false);
		// Sleep to allow the dashboard time to update.
		this_thread::sleep_for(std::chrono::milliseconds(500));
	}
	else
	{
		// Print message if no cameras have been detected.
		cout << "No cameras were detected or no configs have been given from the dashboard." << endl;
	}

	// Close the interpreter and EdgeTPU.
	cout << "Closing Tensorflow Interpreter..." << endl;
	tfliteModelInterpreter.reset();
	cout << "Closing the EdgeTPU device..." << endl;
	edgetpuContext.reset();

	// Print kill message.
	cout << "Program stopped." << endl;
	// Kill program.
	return EXIT_SUCCESS;
}