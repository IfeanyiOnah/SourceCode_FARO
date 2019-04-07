#include "camera_Calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace POL {

	void camera_Calibration::help()
	{
		cout << " camera calibration running..." << endl;
		//	<< "Usage: camera_calibration [configuration_file -- default ./default.xml]" << endl
		//	<< "Near the sample file you'll find the configuration file, which has detailed help of "
		//	"how to edit it.  It may be any OpenCV supported file format XML/YAML." << endl;
	}

	//! [compute_errors]
	double camera_Calibration::computeReprojectionErrors(const vector<vector<cv::Point3f> >& objectPoints,
		const vector<vector<cv::Point2f> >& imagePoints,
		const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
		const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
		vector<float>& perViewErrors, bool fisheye)
	{
		vector<cv::Point2f> imagePoints2;
		size_t totalPoints = 0;
		double totalErr = 0, err;
		perViewErrors.resize(objectPoints.size());

		for (size_t i = 0; i < objectPoints.size(); ++i)
		{
			if (fisheye)
			{
				cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
					distCoeffs);
			}
			else
			{
				projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
			}
			err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

			size_t n = objectPoints[i].size();
			perViewErrors[i] = (float)std::sqrt(err*err / n);
			totalErr += err * err;
			totalPoints += n;
		}

		return std::sqrt(totalErr / totalPoints);
	}
	//! [compute_errors]
	//! [board_corners]
	void camera_Calibration::calcBoardCornerPositions(cv::Size boardSize, float squareSize, vector<cv::Point3f>& corners,
		Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
	{
		corners.clear();

		switch (patternType)
		{
		case Settings::CHESSBOARD:
		case Settings::CIRCLES_GRID:
			for (int i = 0; i < boardSize.height; ++i)
				for (int j = 0; j < boardSize.width; ++j)
					corners.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));
			break;

		case Settings::ASYMMETRIC_CIRCLES_GRID:
			for (int i = 0; i < boardSize.height; i++)
				for (int j = 0; j < boardSize.width; j++)
					corners.push_back(cv::Point3f((2 * j + i % 2)*squareSize, i*squareSize, 0));
			break;
		default:
			break;
		}
	}
	//! [board_corners]
	bool camera_Calibration::runCalibration(Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
		vector<vector<cv::Point2f> > imagePoints, vector<cv::Mat>& rvecs, vector<cv::Mat>& tvecs,
		vector<float>& reprojErrs, double& totalAvgErr)
	{
		//! [fixed_aspect]
		cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
		if (s.flag & cv::CALIB_FIX_ASPECT_RATIO)
			cameraMatrix.at<double>(0, 0) = s.aspectRatio;
		//! [fixed_aspect]
		if (s.useFisheye) {
			distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
		}
		else {
			distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
		}

		vector<vector<cv::Point3f> > objectPoints(1);
		calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

		objectPoints.resize(imagePoints.size(), objectPoints[0]);

		//Find intrinsic and extrinsic camera parameters
		double rms;

		if (s.useFisheye) {
			cv::Mat _rvecs, _tvecs;
			rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
				_tvecs, s.flag);

			rvecs.reserve(_rvecs.rows);
			tvecs.reserve(_tvecs.rows);
			for (int i = 0; i < int(objectPoints.size()); i++) {
				rvecs.push_back(_rvecs.row(i));
				tvecs.push_back(_tvecs.row(i));
			}
		}
		else {
			rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
				s.flag);
		}

		cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

		bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

		totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
			distCoeffs, reprojErrs, s.useFisheye);

		return ok;
	}

	// Print camera parameters to the output file
	void camera_Calibration::saveCameraParams(Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
		const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
		const vector<float>& reprojErrs, const vector<vector<cv::Point2f> >& imagePoints,
		double totalAvgErr, int angle)
	{
		std::stringstream filenam;
		filenam << angle << "_" << s.outputFileName;
		cv::String mystr = filenam.str();
		cv::FileStorage fs(mystr, cv::FileStorage::WRITE);

		time_t tm;
		time(&tm);
		struct tm t2;
		localtime_s(&t2, &tm);
		char buf[1024];
		strftime(buf, sizeof(buf), "%c", &t2);

		fs << "calibration_time" << buf;

		if (!rvecs.empty() || !reprojErrs.empty())
			fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
		fs << "image_width" << imageSize.width;
		fs << "image_height" << imageSize.height;
		fs << "board_width" << s.boardSize.width;
		fs << "board_height" << s.boardSize.height;
		fs << "square_size" << s.squareSize;

		if (s.flag & cv::CALIB_FIX_ASPECT_RATIO)
			fs << "fix_aspect_ratio" << s.aspectRatio;

		if (s.flag)
		{
			std::stringstream flagsStringStream;
			if (s.useFisheye)
			{
				flagsStringStream << "flags:"
					<< (s.flag &cv::fisheye::CALIB_FIX_SKEW ? " +fix_skew" : "")
					<< (s.flag & cv::fisheye::CALIB_FIX_K1 ? " +fix_k1" : "")
					<< (s.flag &cv::fisheye::CALIB_FIX_K2 ? " +fix_k2" : "")
					<< (s.flag &cv::fisheye::CALIB_FIX_K3 ? " +fix_k3" : "")
					<< (s.flag & cv::fisheye::CALIB_FIX_K4 ? " +fix_k4" : "")
					<< (s.flag & cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ? " +recompute_extrinsic" : "");
			}
			else
			{
				flagsStringStream << "flags:"
					<< (s.flag & cv::CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
					<< (s.flag &cv::CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
					<< (s.flag & cv::CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
					<< (s.flag & cv::CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
					<< (s.flag &cv::CALIB_FIX_K1 ? " +fix_k1" : "")
					<< (s.flag &cv::CALIB_FIX_K2 ? " +fix_k2" : "")
					<< (s.flag & cv::CALIB_FIX_K3 ? " +fix_k3" : "")
					<< (s.flag & cv::CALIB_FIX_K4 ? " +fix_k4" : "")
					<< (s.flag & cv::CALIB_FIX_K5 ? " +fix_k5" : "");
			}
			fs.writeComment(flagsStringStream.str());
		}

		fs << "flags" << s.flag;

		fs << "fisheye_model" << s.useFisheye;

		fs << "camera_matrix" << cameraMatrix;
		fs << "distortion_coefficients" << distCoeffs;

		fs << "avg_reprojection_error" << totalAvgErr;
		if (s.writeExtrinsics && !reprojErrs.empty())
			fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);

		if (s.writeExtrinsics && !rvecs.empty() && !tvecs.empty())
		{
			CV_Assert(rvecs[0].type() == tvecs[0].type());
			cv::Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
			bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
			bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

			for (size_t i = 0; i < rvecs.size(); i++)
			{
				cv::Mat r = bigmat(cv::Range(int(i), int(i + 1)), cv::Range(0, 3));
				cv::Mat t = bigmat(cv::Range(int(i), int(i + 1)), cv::Range(3, 6));

				if (needReshapeR)
					rvecs[i].reshape(1, 1).copyTo(r);
				else
				{
					//*.t() is MatExpr (not cv::Mat) so we can use assignment operator
					CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
					r = rvecs[i].t();
				}

				if (needReshapeT)
					tvecs[i].reshape(1, 1).copyTo(t);
				else
				{
					CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
					t = tvecs[i].t();
				}
			}
			fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
			fs << "extrinsic_parameters" << bigmat;
		}

		if (s.writePoints && !imagePoints.empty())
		{
			cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
			for (size_t i = 0; i < imagePoints.size(); i++)
			{
				cv::Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
				cv::Mat imgpti(imagePoints[i]);
				imgpti.copyTo(r);
			}
			fs << "image_points" << imagePtMat;
		}
	}

	//! [run_and_save]
	bool camera_Calibration::runCalibrationAndSave(Settings& s, cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
		vector<vector<cv::Point2f> > imagePoints, int angle)
	{
		vector<cv::Mat> rvecs, tvecs;
		vector<float> reprojErrs;
		double totalAvgErr = 0;

		bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,
			totalAvgErr);
		cout << (ok ? "Calibration succeeded" : "Calibration failed")
			<< ". avg re projection error = " << totalAvgErr << endl;
		s.imHeight = imageSize.height;
		s.imWidth = imageSize.width;
		myParameters()->imgHeight = s.imHeight;
		myParameters()->imgWidth = s.imWidth;
		if (ok)
			switch (angle)
			{
			case 0:
				myParameters()->cameraMatrix0 = cameraMatrix.clone();
				myParameters()->distCoeffs0 = distCoeffs.clone();
				break;
			case 45:
				myParameters()->cameraMatrix45 = cameraMatrix.clone();
				myParameters()->distCoeffs45 = distCoeffs.clone();
				break;
			case 90:
				myParameters()->cameraMatrix90 = cameraMatrix.clone();
				myParameters()->distCoeffs90 = distCoeffs.clone();
				break;
			case 135:
				myParameters()->cameraMatrix135 = cameraMatrix.clone();
				myParameters()->distCoeffs135 = distCoeffs.clone();
				break;
			default:
				break;
			}
		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
			totalAvgErr, angle);
		return ok;
	}
	//! [run_and_save]



	int camera_Calibration::Calibrate(const string &inputSettingsFile) {
		help();
		//! [file_read]
		Settings s;
		////const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
		//cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
		//if (!fs.isOpened())
		//{
		//	cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
		//	return -1;
		//}
		//fs["polarization_settings.camera_calibration"] >> s;
		//fs.release();                                         // close Settings file
		//													  //! [file_read]

		//													  //cv::FileStorage fout("settings.yml", cv::FileStorage::WRITE); // write config as YAML
		//													  //fout << "Settings" << s;


		//! [get_input]
		s.setUp(inputSettingsFile);


		if (!s.goodInput)
		{
			error("Invalid input detected. Application stopping. ");
			return -1;
		}


		std::vector<std::vector<cv::Mat>>img;
		img.resize(4);
		vector<cv::Mat> tmp;
		for (int i = 0; i < s.nrFrames; ++i) {
			cv::Mat view = s.nextImage().clone();
			cv::cvtColor(view, view, CV_BGR2GRAY);
			imgSplit(view, tmp, s.v_raw_sep);
			img[0].push_back(tmp[0].clone());
			img[1].push_back(tmp[1].clone());
			img[2].push_back(tmp[2].clone());
			img[3].push_back(tmp[3].clone());
		}


		for (int frameNum = 0, angle = 0; frameNum < img.size(); angle += 45, ++frameNum) {
			bool blinkOutput = false;
			vector<vector<cv::Point2f> > imagePoints;
			cv::Mat  cameraMatrix, distCoeffs;
			cv::Size imageSize;
			int mode = s.inputType == Settings::IMAGE_LIST ? camera_Calibration::CAPTURING : camera_Calibration::DETECTION;
			clock_t prevTimestamp = 0;
			const cv::Scalar RED(0, 0, 255), GREEN(0, 255, 0);
			const char ESC_KEY = 27;


			//for (int x = 0; x < img[frameNum].size(); ++x)
			for (;;)
			{
				cv::Mat view;
				if (!img[frameNum].empty()) {
					view = img[frameNum].at(img[frameNum].size() - 1).clone();
					img[frameNum].pop_back();
				}

				//-----  If no more image, or got enough, then stop calibration and show result -------------
				if (mode == camera_Calibration::CAPTURING && imagePoints.size() >= (size_t)s.nrFrames)
				{
					if (runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints, angle))
						if (frameNum == img.size() - 1)mode = camera_Calibration::CALIBRATED;
						else
							mode = camera_Calibration::DETECTION;
				}
				if (view.empty())          // If there are no more images stop the loop
				{
					// if calibration threshold was not reached yet, calibrate now
					if (mode != camera_Calibration::CALIBRATED && !imagePoints.empty())
						runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints, angle);
					break;
				}
				//! [get_input]

				imageSize = view.size();  // Format input image.
				if (s.flipVertical)    flip(view, view, 0);

				//! [find_pattern]
				vector<cv::Point2f> pointBuf;

				bool found;

				int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

				if (!s.useFisheye) {
					// fast check erroneously fails with high distortions like fisheye
					chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
				}

				switch (s.calibrationPattern) // Find feature points on the input format
				{
				case Settings::CHESSBOARD:
					found = findChessboardCorners(view, s.boardSize, pointBuf, chessBoardFlags);
					break;
				case Settings::CIRCLES_GRID:
					found = findCirclesGrid(view, s.boardSize, pointBuf);
					break;
				case Settings::ASYMMETRIC_CIRCLES_GRID:
					found = findCirclesGrid(view, s.boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID);
					break;
				default:
					found = false;
					break;
				}
				//! [find_pattern]
				//! [pattern_found]
				if (found)                // If done with success,
				{
					// improve the found corners' coordinate accuracy for chessboard
					if (s.calibrationPattern == Settings::CHESSBOARD)
					{
						cv::Mat viewGray = view.clone();

						//cvtColor(view, viewGray, cv::COLOR_BGR2GRAY);
						cornerSubPix(viewGray, pointBuf, cv::Size(11, 11),
							cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
					}

					if (mode == camera_Calibration::CAPTURING &&  // For camera only take new samples after delay time
						(!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC))
					{
						imagePoints.push_back(pointBuf);
						prevTimestamp = clock();
						blinkOutput = s.inputCapture.isOpened();
					}

					// Draw the corners.
					drawChessboardCorners(view, s.boardSize, cv::Mat(pointBuf), found);
				}
				//! [pattern_found]
				//----------------------------- Output Text ------------------------------------------------
				//! [output_text]
				string msg = (mode == camera_Calibration::CAPTURING) ? "100/100" :
					mode == camera_Calibration::CALIBRATED ? "Calibrated" : "Press 'g' to start";
				int baseLine = 0;
				cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseLine);
				cv::Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

				if (mode == camera_Calibration::CAPTURING)
				{
					if (s.showUndistorsed)
						msg = cv::format("%d/%d Undist", (int)imagePoints.size(), s.nrFrames);
					else
						msg = cv::format("%d/%d", (int)imagePoints.size(), s.nrFrames);
				}

				putText(view, msg, textOrigin, 1, 1, mode == camera_Calibration::CALIBRATED ? GREEN : RED);

				if (blinkOutput)
					bitwise_not(view, view);
				//! [output_text]
				//------------------------- Video capture  output  undistorted ------------------------------
				//! [output_undistorted]
				if (mode == camera_Calibration::CALIBRATED && s.showUndistorsed)
				{
					cv::Mat temp = view.clone();
					if (s.useFisheye)
						cv::fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs);
					else
						undistort(temp, view, cameraMatrix, distCoeffs);
				}
				////! [output_undistorted]
				////------------------------------ Show image and check for input commands -------------------
				////! [await_input]
				//cv::namedWindow("Image View", CV_WINDOW_NORMAL);
				//cv::imshow("Image View", view);
				//char key = (char)cv::waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

				//if (key == ESC_KEY)
				//	break;

				//if (key == 'u' && mode == CALIBRATED)
				//	s.showUndistorsed = !s.showUndistorsed;

				//if (s.inputCapture.isOpened() && key == 'g')
				//{
				//	mode = CAPTURING;
				//	imagePoints.clear();
				//}
				////! [await_input]
			}


		}

		return 0;

	}

	cv::Mat camera_Calibration::imgUndist(cv::Mat view, cv::Mat cameraMatrix, cv::Mat distCoef) const {

		cv::Mat rview, map1, map2;

		cv::Size imageSize = cv::Size(view.cols, view.rows);

		cameraMatrix.convertTo(cameraMatrix, CV_32FC1);
		distCoef.convertTo(distCoef, CV_32FC1);
		cv::Mat tmp;
		view.convertTo(tmp, CV_32FC1);

		initUndistortRectifyMap(cameraMatrix, distCoef, cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix, distCoef, imageSize, 1, imageSize, 0), imageSize, CV_32FC1, map1, map2);

		remap(tmp, rview, map1, map2, cv::INTER_LINEAR);

		//view = view / 100.f;
		rview.convertTo(rview, view.type());

		return rview.clone();
	}



}