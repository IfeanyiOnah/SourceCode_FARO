#include "Settings.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include "polarizationBase.h"
namespace POL {

	Settings::Settings() : goodInput(false)
	{

	}


	Settings::~Settings()
	{
	}

	void Settings::write(cv::FileStorage& fs) const                        //Write serialization for this class
	{
		fs << "{"
			<< "BoardSize_Width" << boardSize.width
			<< "BoardSize_Height" << boardSize.height
			<< "Square_Size" << squareSize
			<< "Calibrate_Pattern" << patternToUse
			<< "Calibrate_NrOfFrameToUse" << nrFrames
			<< "Calibrate_FixAspectRatio" << aspectRatio
			<< "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
			<< "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

			<< "Write_DetectedFeaturePoints" << writePoints
			<< "Write_extrinsicParameters" << writeExtrinsics
			<< "Write_outputFileName" << outputFileName

			<< "Show_UndistortedImage" << showUndistorsed

			<< "Input_FlipAroundHorizontalAxis" << flipVertical
			<< "Input_Delay" << delay
			<< "Input" << input
			<< "}";
	}
	void Settings::read(const cv::FileNode& node)                          //Read serialization for this class
	{
		node["BoardSize_Width"] >> boardSize.width;
		node["BoardSize_Height"] >> boardSize.height;
		node["Calibrate_Pattern"] >> patternToUse;
		node["Square_Size"] >> squareSize;
		node["Calibrate_NrOfFrameToUse"] >> nrFrames;
		node["Calibrate_FixAspectRatio"] >> aspectRatio;
		node["Write_DetectedFeaturePoints"] >> writePoints;
		node["Write_extrinsicParameters"] >> writeExtrinsics;
		node["Write_outputFileName"] >> outputFileName;
		node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
		node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
		node["Calibrate_UseFisheyeModel"] >> useFisheye;
		node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
		node["Show_UndistortedImage"] >> showUndistorsed;
		node["Input"] >> input;
		node["Input_Delay"] >> delay;
		node["Fix_K1"] >> fixK1;
		node["Fix_K2"] >> fixK2;
		node["Fix_K3"] >> fixK3;
		node["Fix_K4"] >> fixK4;
		node["Fix_K5"] >> fixK5;

		node["image_width"] >> imWidth;
		node["image_height"] >> imHeight;
		node["camera_matrix"] >> cameraMatrix;
		node["distortion_coefficients"] >> distCoef;

		validate();
	}
	void Settings::validate()
	{
		goodInput = true;
		if (boardSize.width <= 0 || boardSize.height <= 0)
		{
			cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
			goodInput = false;
		}
		if (squareSize <= 10e-6)
		{
			cerr << "Invalid square size " << squareSize << endl;
			goodInput = false;
		}
		if (nrFrames <= 0)
		{
			cerr << "Invalid number of frames " << nrFrames << endl;
			goodInput = false;
		}
		if (!(v_raw_sep == polarizationBase::RAW || v_raw_sep == polarizationBase::SEPARATED))
		{
			error("invalid calibration image type: ", "Please enter 'raw' or 'sep' at the settings file");
			goodInput = false;
		}

		if (input.empty())      // Check for valid input
			inputType = INVALID;
		else
		{
			if (input[0] >= '0' && input[0] <= '9')
			{
				stringstream ss(input);
				ss >> cameraID;
				inputType = CAMERA;
			}
			else
			{
				if (isListOfImages(input) && readStringList(input, imageList))
				{
					inputType = IMAGE_LIST;
					nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
				}
				else
					inputType = VIDEO_FILE;
			}
			if (inputType == CAMERA)
				inputCapture.open(cameraID);
			if (inputType == VIDEO_FILE)
				inputCapture.open(input);
			if (inputType != IMAGE_LIST && !inputCapture.isOpened())
				inputType = INVALID;
		}
		if (inputType == INVALID)
		{
			cerr << " Input does not exist: " << input;
			goodInput = false;
		}

		flag = 0;
		if (calibFixPrincipalPoint) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
		if (calibZeroTangentDist)   flag |= cv::CALIB_ZERO_TANGENT_DIST;
		if (aspectRatio)            flag |= cv::CALIB_FIX_ASPECT_RATIO;
		if (fixK1)                  flag |= cv::CALIB_FIX_K1;
		if (fixK2)                  flag |= cv::CALIB_FIX_K2;
		if (fixK3)                  flag |= cv::CALIB_FIX_K3;
		if (fixK4)                  flag |= cv::CALIB_FIX_K4;
		if (fixK5)                  flag |= cv::CALIB_FIX_K5;

		if (useFisheye) {
			// the fisheye model has its own enum, so overwrite the flags
			flag = cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
			if (fixK1)                   flag |= cv::fisheye::CALIB_FIX_K1;
			if (fixK2)                   flag |= cv::fisheye::CALIB_FIX_K2;
			if (fixK3)                   flag |= cv::fisheye::CALIB_FIX_K3;
			if (fixK4)                   flag |= cv::fisheye::CALIB_FIX_K4;
			if (calibFixPrincipalPoint) flag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
		}

		calibrationPattern = NOT_EXISTING;
		if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
		if (!patternToUse.compare("CIRCLES_GRID")) calibrationPattern = CIRCLES_GRID;
		if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID")) calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
		if (calibrationPattern == NOT_EXISTING)
		{
			cerr << " Camera calibration mode does not exist: " << patternToUse << endl;
			goodInput = false;
		}
		atImageList = 0;

	}
	cv::Mat Settings::nextImage()
	{
		cv::Mat result;
		if (inputCapture.isOpened())
		{
			cv::Mat view0;
			inputCapture >> view0;
			view0.copyTo(result);
		}
		else if (atImageList < imageList.size())
			result = cv::imread(imageList[atImageList++], cv::IMREAD_COLOR);

		return result;
	}

	bool Settings::readStringList(const string& filename, vector<string>& l)
	{
		// Create empty property tree object
		namespace pt = boost::property_tree;
		pt::ptree tree;

		// Parse the XML into the property tree.
		pt::read_xml(filename, tree);
		l.clear();
		//cv::FileStorage fs(filename, cv::FileStorage::READ);
		//if (!fs.isOpened())
		//	return false;
		//cv::FileNode n = fs.getFirstTopLevelNode();
		//if (n.type() != cv::FileNode::SEQ)
		//	return false;
		//cv::FileNodeIterator it = n.begin(), it_end = n.end();
		//for (; it != it_end; ++it)
		//	l.push_back((string)*it);
				// Parse the XML into the property tree.
		pt::read_xml(filename, tree);
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v, tree.get_child("opencv_storage.Images"))
			l.push_back(v.second.data());

		return true;
	}

	bool Settings::isListOfImages(const string& filename)
	{
		string s(filename);
		// Look for file extension
		if (s.find(".xml") == string::npos && s.find(".yaml") == string::npos && s.find(".yml") == string::npos)
			return false;
		else
			return true;
	}

	void  Settings::setUp(const string &filename) {
		// Create empty property tree object
		namespace pt = boost::property_tree;
		pt::ptree tree;
		
		// Parse the XML into the property tree.
		pt::read_xml(filename, tree);

		// Use get_child to find the node containing the modules, and iterate over
		// its children. If the path cannot be resolved, get_child throws.
		// A C++11 for-range loop would also work.
		int widt = 0, height = 0;
		BOOST_FOREACH(pt::ptree::value_type &v, tree.get_child("polarization_settings.camera_calibration")) {
			// The data function is used to access the data stored in a node.
			if (v.first == "BoardSize_Width")stringstream(v.second.data()) >> widt;
			else if (v.first == "BoardSize_Height")stringstream(v.second.data()) >> height;
			else if (v.first == "Square_Size")stringstream(v.second.data()) >> squareSize;
			else if (v.first == "Calibrate_Pattern")stringstream(v.second.data()) >> patternToUse;
			else if (v.first == "Input") input = v.second.data();
			else if (v.first == "Input_FlipAroundHorizontalAxis")stringstream(v.second.data()) >> flipVertical;
			else if (v.first == "Input_Delay")stringstream(v.second.data()) >> delay;
			else if (v.first == "Calibrate_NrOfFrameToUse")stringstream(v.second.data()) >> nrFrames;
			else if (v.first == "Calibrate_FixAspectRatio")stringstream(v.second.data()) >> aspectRatio;
			else if (v.first == "Calibrate_AssumeZeroTangentialDistortion")stringstream(v.second.data()) >> calibZeroTangentDist;
			else if (v.first == "Calibrate_FixPrincipalPointAtTheCenter")stringstream(v.second.data()) >> calibFixPrincipalPoint;
			else if (v.first == "Write_outputFileName")stringstream(v.second.data()) >> outputFileName;
			else if (v.first == "Write_DetectedFeaturePoints")stringstream(v.second.data()) >> writePoints;
			else if (v.first == "Write_extrinsicParameters")stringstream(v.second.data()) >> writeExtrinsics;
			else if (v.first == "Show_UndistortedImage")stringstream(v.second.data()) >> showUndistorsed;
			else if (v.first == "Calibrate_UseFisheyeModel")stringstream(v.second.data()) >> useFisheye;
			else if (v.first == "Fix_K1")stringstream(v.second.data()) >> fixK1;
			else if (v.first == "Fix_K2")stringstream(v.second.data()) >> fixK2;
			else if (v.first == "Fix_K3")stringstream(v.second.data()) >> fixK3;
			else if (v.first == "Fix_K4")stringstream(v.second.data()) >> fixK4;
			else if (v.first == "Fix_K5")stringstream(v.second.data()) >> fixK5;
			else if (v.first == "Raw_Sep")stringstream(v.second.data()) >> v_raw_sep;
		}

		boardSize = cv::Size(widt, height);	
	
		validate();

	}

}