#include "pcdFilterPolarization.h"
#include"Demosaic.h"
#include "camera_Calibration.h"
#include "polarization_Parameters.h"
#include "Transformation.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
namespace pt = boost::property_tree;


namespace POL {

	void pcdFilterPolarization::filter(int argv, char **argc) {

		//check if the input arguement is passed
		if (argv != 2) {
			error("Error: Invalid input file: filter()");
		}

		//initialize the algorithm from settings file
		setUp(argc[1]);

		//read the image file and demosaic the image
		Demosaic demo(v_demosaic_method);
		imgDemosaic(demo);

		//caliberate camera
		camera_Calibration calib;
		camCalibration(argc[1], calib);

		//undistort the image
		exeUndist(demo,calib);

		//compute polarization parameters
			// instantiate the object of the Polarization class
		polarization_Parameters objPol(v_surface_type);
		polParameters(objPol,demo);

		//read the object point and the image point and compute the extrinsic camera matrix
		//Also check for normal vector computation and project the point to image plane
		Transformation objTrans(v_radius_size);
		exePointProjection(objTrans,demo, calib);

		//compute the meta data
		calcMetaData(objPol, objTrans);

		//generate mask from the computed meta data using stdErrMean
		segmentMultireflection(metaData, segmentedOutput, v_stdErrMeanKernSize, v_stdErrMeanVal);

		//apply morphology operation: dilation function from opencv library
		cv::dilate(segmentedOutput, morphOutput, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, 
			cv::Size(v_morphKernSize, v_morphKernSize)));

		//remove the points in the point cloud detected by the mask
		TrimPointCloud(objPol, objTrans);

		//save the filtered point cloud as well as the mask revealing the areas affected by multireflection
		SaveMetaData(objTrans);
	}

	void pcdFilterPolarization::segmentMultireflection(const cv::Mat &input, cv::Mat &output, int maskSize, double stdErrMean) {
		//test for the validity of the input 
		if (input.empty())error("Error: input is not valid: ", "segmentMultireflection()");
		if (stdErrMean < 0)error("Error: invalid value of standard error of the mean: ", "segmentMultireflection()");

		//test for the validity of the mask size
		if (maskSize <= 0)error("Error: invalid mask size passed to segmentMultireflection(): ", maskSize);
		if (maskSize > input.rows || maskSize > input.cols) {
			cerr << "Invalid mask size: " << maskSize << " image size: " << input.size() << endl;
			error("the mask size is greater than the input image size");
		}


		//we ensure that the mask size is always even
		//this helps to compute the offset if the mask size is not a factor of image size 
		if ((maskSize % 2) != 0) ++maskSize;

		//get the offset from the mask and the pad
		int offsety = input.rows - (input.rows / maskSize * maskSize);
		int offsetx = input.cols - (input.cols / maskSize * maskSize);

		if (offsety < 0 || offsetx < 0)error("Error: invalid kernel size for the standard error of the mean: ", maskSize);
		//compute size of the pad to enable the mask size divisible by image size
		int pady = (maskSize - offsety) / 2;
		int padx = (maskSize - offsetx) / 2;

		//pad the input and output image
		cv::Mat tmp, inputPadded;

		tmp = input.clone();
		tmp.convertTo(tmp, CV_64FC1);

		padImage(tmp, inputPadded, padx, pady);

		int rows = inputPadded.rows;
		int cols = inputPadded.cols;

		cv::Mat outputPadded(rows, cols, inputPadded.type());

		//instantiate the object of the local mask with same size as the given masksize
		cv::Mat mask(maskSize, maskSize, inputPadded.type(), cv::Scalar::all(0.0));

		//loop over the padded input image
		for (int r = 0; r < rows; r += maskSize) {
			for (int c = 0; c < cols; c += maskSize) {
				//get the local region same size with the masksize
				cv::Mat getLocalregion(maskSize, maskSize, inputPadded.type());
				inputPadded(cv::Rect(c, r, maskSize, maskSize)).copyTo(getLocalregion);

				//compute the standard deviation
				cv::Scalar mean, std;
				cv::meanStdDev(getLocalregion, mean, std);

				//compute the standard error of the mean
				double err = std[0] / sqrt(pow(double(maskSize), 2.0));

				//check if the computed value is below the global value
				if (err > stdErrMean) {
					double mini, maxi;
					while (err > stdErrMean)
						//if the current err is above the local value then loop over the local image
					{
						//get the maximum and minimum value within the local region
						cv::minMaxIdx(getLocalregion, &mini, &maxi);
						for (int i = 0; i < maskSize; ++i) {
							double *PgetLocalregion = getLocalregion.ptr<double>(i);
							double *Pmask = mask.ptr<double>(i);
							for (int j = 0; j < maskSize; ++j) {
								//all point equal to maximum will be assigned minimum
								if (maxi == *PgetLocalregion) {
									*PgetLocalregion = mini;

									//assign 1 to the value of mask at that region(s) 
									*Pmask = 1;
								}
								++PgetLocalregion;
								++Pmask;
							}
						}
						//recompute the standard error of the mean and check again if it is less than global value
						cv::meanStdDev(getLocalregion, mean, std);
						err = std[0] / sqrt(pow(double(maskSize), 2.0));

					}
				}
				//copy the mask back to the output image buffer
				mask.copyTo(outputPadded(cv::Rect(c, r, maskSize, maskSize)));

				//reinitialize the mask value
				mask = cv::Scalar::all(0.0);
			}
		}

		output.release();
		output = cv::Mat(input.rows, input.cols, input.type()).clone();
		//return the padded output to the original image size 
		outputPadded.convertTo(outputPadded, input.type());
		outputPadded(cv::Rect(padx, pady, input.cols, input.rows)).copyTo(output);
		//cv::multiply(input, output, output);

		//test for the validity of the input 
		if (output.empty())error("Error: output is not valid: ", "segmentMultireflection()");
	}

	void pcdFilterPolarization::imgDemosaic(Demosaic &dem) {
	
		//declare variable for the image buffer and read image from file
		Mat Img_gray;
		Mat data = imread(v_imgFileName);

		//convert image to gray
		cvtColor(data, Img_gray, CV_BGR2GRAY);

		//set the input image
		dem.myParameters()->Input = Img_gray.clone();

		//set the polarization orientation
		dem.myParameters()->orientPolAngleR0C0 = v_orientPolAngleR0C0;
		dem.myParameters()->orientPolAngleR1C0 = v_orientPolAngleR1C0;
		dem.myParameters()->orientPolAngleR0C1 = v_orientPolAngleR0C1;
		dem.myParameters()->orientPolAngleR1C1 = v_orientPolAngleR1C1;

		//execute the algorithm
		dem.exeDemosaic();

		imwrite("Angle0.png", dem.myParameters()->Output.at(0));
		imwrite("Angle45.png", dem.myParameters()->Output.at(1));
		imwrite("Angle90.png", dem.myParameters()->Output.at(2));
		imwrite("Angle135.png", dem.myParameters()->Output.at(3));
	}


	void pcdFilterPolarization::camCalibration(const string &filename, camera_Calibration& calib) {
		//if calib is false then load the calibration parameter from the file
		if (!v_calibrateCam)
			loadCameraParameters(calib);
		else
			calib.Calibrate(filename);	//execute the calibration
	}

	void pcdFilterPolarization::loadCameraParameters(camera_Calibration& calib) {
		//read the already calibrated output file
		cv::FileStorage fs;
		Settings s;
		string filename;

		filename = "0_" + v_outputFileName;
		fs.open(filename, cv::FileStorage::READ); // Read the settings
		//get the image with and height
		fs["image_width"] >> calib.myParameters()->imgWidth;
		fs["image_height"] >> calib.myParameters()->imgHeight;
		if (!fs.isOpened())
		{
			error("Could not open the configuration file 0_camera_parameters.xml");

		}
		fs["camera_matrix"] >> calib.myParameters()->cameraMatrix0;
		fs["distortion_coefficients"] >> calib.myParameters()->distCoeffs0;
		fs.release();

		filename = "45_" + v_outputFileName;
		fs.open(filename, cv::FileStorage::READ); // Read the settings
		if (!fs.isOpened())
		{
			error("Could not open the configuration file 45_camera_parameters.xml");
		}
		fs["camera_matrix"] >> calib.myParameters()->cameraMatrix45;
		fs["distortion_coefficients"] >> calib.myParameters()->distCoeffs45;
		fs.release();

		filename = "90_" + v_outputFileName;
		fs.open(filename, cv::FileStorage::READ); // Read the settings
		if (!fs.isOpened())
		{
			error("Could not open the configuration file 90_camera_parameters.xml");
		}
		fs["camera_matrix"] >> calib.myParameters()->cameraMatrix90;
		fs["distortion_coefficients"] >> calib.myParameters()->distCoeffs90;
		fs.release();

		filename = "135_" + v_outputFileName;
		fs.open(filename, cv::FileStorage::READ); // Read the settings
		if (!fs.isOpened())
		{
			error("Could not open the configuration file 135_camera_parameters.xml");
		}
		fs["camera_matrix"] >> calib.myParameters()->cameraMatrix135;
		fs["distortion_coefficients"] >> calib.myParameters()->distCoeffs135;
		fs.release();
	}

	void pcdFilterPolarization::exeUndist(Demosaic& dem, const camera_Calibration& calib) {
	
		if (dem.myParameters()->Output.empty())
			error("invalid output image buffer for demosaic container: ");

		string er = "Please check the demosaic type at the settings file or use same image size";

		if (dem.myParameters()->Output[0].size() !=
			cv::Size(calib.myParameters()->imgWidth, calib.myParameters()->imgHeight))
			error("Error: image size after calibration not equals image size after demosaic:\n ", er);

		for (int i = 0; i <= 135; i += 45) {
			//undist image
			switch (i) {
			case 0:
				dem.myParameters()->Output[0] = calib.imgUndist(dem.myParameters()->Output[0],
				calib.myParameters()->cameraMatrix0, calib.myParameters()->distCoeffs0);
				break;
			case 45:
				dem.myParameters()->Output[1] = calib.imgUndist(dem.myParameters()->Output[1], 
				calib.myParameters()->cameraMatrix45, calib.myParameters()->distCoeffs45);
				break;
			case 90:
				dem.myParameters()->Output[2] = calib.imgUndist(dem.myParameters()->Output[2], 
				calib.myParameters()->cameraMatrix90, calib.myParameters()->distCoeffs90);
				break;
			case 135:
				dem.myParameters()->Output[3] = calib.imgUndist(dem.myParameters()->Output[3],
				calib.myParameters()->cameraMatrix135, calib.myParameters()->distCoeffs135);
				break;
			default:
				break;
			}
		}


	}


	void pcdFilterPolarization::polParameters(polarization_Parameters& pol,const Demosaic &dem) {
	
		//declare variable for the image buffer and read image from file
		//Mat data = imread("im.tif");

		//set the input image
		pol.myParameters()->Input = dem.myParameters()->Output;

		//set refractive index
		pol.myParameters()->refind = v_refind;

		//set the polarization orientation
		pol.myParameters()->orientPolAngleR0C0 = v_orientPolAngleR0C0;
		pol.myParameters()->orientPolAngleR1C0 = v_orientPolAngleR1C0;
		pol.myParameters()->orientPolAngleR0C1 = v_orientPolAngleR0C1;
		pol.myParameters()->orientPolAngleR1C1 = v_orientPolAngleR1C1;

		//execute the algorithm
		pol.executePolarizationParameters();
	}

	void pcdFilterPolarization::exePointProjection(Transformation& Trans, const Demosaic &dem, 
		const camera_Calibration& calib)
	{

		//compute rotational and translational vector
		Trans.getRotTrans(v_imgPointFileName, v_objPointFileName,
			calib.myParameters()->cameraMatrix0, calib.myParameters()->distCoeffs0,
			Trans.myParameters()->rotv, Trans.myParameters()->tranv);

		//load obejct points
		Trans.loadFilePCD(v_pcdFileName, Trans.myParameters()->original, v_calc_norm);

		//std::cout << "\nfinish loading 3D points... " << endl;

		//project the point to the image
		Trans.ProjectPoint3D(Trans.myParameters()->original,
			Trans.myParameters()->projected, calib.myParameters()->cameraMatrix0,
			Trans.myParameters()->rotv, Trans.myParameters()->tranv);

		//std::cout << "\nfinishing projecting 3D points: " << endl;

		// filter along z-dimension while looking at the camera viewing direction 
		Trans.camViewFilterZdim(dem.myParameters()->Output[0].rows,
			dem.myParameters()->Output[0].cols);
	}

	void  pcdFilterPolarization::calcAngDiff(polarization_Parameters& pol, Transformation& Trans) {

		//get the equation that depends on angular differences
		cv::Mat polx, poly, polz, rnx, rny, rnz;

		polx = pol.myParameters()->Normal_x.clone();
		poly = pol.myParameters()->Normal_y.clone();
		polz = pol.myParameters()->Normal_z.clone();

		rnx = Trans.myParameters()->rotNx.clone();
		rny = Trans.myParameters()->rotNy.clone();
		rnz = Trans.myParameters()->rotNz.clone();
		rnx.convertTo(rnx, polx.type());
		rny.convertTo(rny, polx.type());
		rnz.convertTo(rnz, polx.type());

		int rows = polx.rows;
		int cols = polx.cols;

		cv::Mat tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;

		angDiff.create(rows, cols, polx.type());

		//calc the angle between the vector
		cv::multiply(rnx, polx, tmp1);
		cv::multiply(rny, poly, tmp2);
		cv::multiply(rnz, polz, tmp3);
		cv::add(tmp1, tmp2, tmp4);
		cv::add(tmp4, tmp3, tmp4);

		cv::pow(rnx, 2.0, tmp1);
		cv::pow(rny, 2.0, tmp2);
		cv::pow(rnz, 2.0, tmp3);
		cv::add(tmp1, tmp2, tmp5);
		cv::add(tmp5, tmp3, tmp5);
		cv::sqrt(tmp5, magN);

		cv::pow(polx, 2.0, tmp1);
		cv::pow(poly, 2.0, tmp2);
		cv::pow(polz, 2.0, tmp3);
		cv::add(tmp1, tmp2, tmp6);
		cv::add(tmp6, tmp3, tmp6);
		cv::sqrt(tmp6, magG);

		cv::multiply(magN, magG, tmp5);
		cv::divide(tmp4, tmp5, tmp6);

		tmp6.convertTo(tmp6, CV_32F);
		cv::patchNaNs(tmp6, 0.f);
		tmp6.convertTo(tmp6, CV_64F);

		for (int r = 0; r < rows; ++r) {
			double *Ptmp = tmp6.ptr<double>(r);
			double *Pang = angDiff.ptr<double>(r);
			for (int c = 0; c < cols; ++c) {
				double res = std::acos(abs(*Ptmp));
				*Pang = res;
				++Pang;
				++Ptmp;
			}
		}
		angDiff.convertTo(angDiff, CV_32F);
		cv::patchNaNs(angDiff, 0.f);
		angDiff.convertTo(angDiff, CV_64F);
	}

	void   pcdFilterPolarization::calcMagDiff(polarization_Parameters& pol, Transformation& Trans) {
		//scale DoLP
		cv::Mat zenith, intensity;
		pol.myParameters()->Intensity.convertTo(intensity, pol.myParameters()->DoLP.type());
		cv::multiply(intensity / 35.0, pol.myParameters()->DoLP, scaleDoLP);

		//recompute the normal vector from polarization
		pol.calcZenith(scaleDoLP, zenith);

		//recompute the normal vector from polarization
		pol.calcNormalVectorPol(zenith, pol.myParameters()->Azimuth);

		//get the equation that depends on magnitude differences
		cv::Mat polx, poly, polz, rnx, rny, rnz;

		polx = pol.myParameters()->Normal_x.clone();
		poly = pol.myParameters()->Normal_y.clone();
		polz = pol.myParameters()->Normal_z.clone();

		rnx = Trans.myParameters()->rotNx.clone();
		rny = Trans.myParameters()->rotNy.clone();
		rnz = Trans.myParameters()->rotNz.clone();
		rnx.convertTo(rnx, polx.type());
		rny.convertTo(rny, polx.type());
		rnz.convertTo(rnz, polx.type());

		int rows = polx.rows;
		int cols = polx.cols;

		cv::Mat tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;

		magDiff.create(rows, cols, polx.type());

		//calc the angle between the vector
		cv::divide(rnx, rnz, rnx);
		cv::divide(rny, rnz, rny);
		cv::pow(rnx, 2.0, tmp1);
		cv::pow(rny, 2.0, tmp2);
		cv::add(tmp1, tmp2, tmp5);
		cv::sqrt(tmp5, magN);

		cv::divide(polx, polz, polx);
		cv::divide(poly, polz, poly);
		cv::pow(polx, 2.0, tmp1);
		cv::pow(poly, 2.0, tmp2);
		cv::add(tmp1, tmp2, tmp5);
		cv::sqrt(tmp5, magG);

		tmp1 = abs(magN - magG);
		tmp2 = magN + magG;
		cv::divide(tmp1, tmp2, magDiff);
		magDiff = 1 - magDiff;

		magDiff.convertTo(magDiff, CV_32F);
		magG.convertTo(magG, CV_32F);
		magN.convertTo(magN, CV_32F);

		cv::patchNaNs(magDiff, 0.f);
		cv::patchNaNs(magG, 0.f);
		cv::patchNaNs(magN, 0.f);

		magDiff.convertTo(magDiff, CV_64F);
		magG.convertTo(magG, CV_64F);
		magN.convertTo(magN, CV_64F);


		for (int r = 0; r < rows; ++r) {
			double *PmagDiff = magDiff.ptr<double>(r);
			double *PmagG = magG.ptr<double>(r);
			double *PmagN = magN.ptr<double>(r);
			for (int c = 0; c < cols; ++c) {
				if (*PmagG == 0 || *PmagN == 0)*PmagDiff = 0;
				++PmagDiff;
				++PmagN;
				++PmagG;
			}
		}


	}

	void  pcdFilterPolarization::calcMagAngDiff()
	{
		cv::multiply(magDiff, angDiff, magAng);
	}

	void pcdFilterPolarization::calcMetaData(polarization_Parameters& pol, Transformation& Trans) {

		//calc angular diff
		calcAngDiff(pol,Trans);

		//calc magnitude diff
		calcMagDiff(pol,Trans);

		//calc angular-magnitude diff
		calcMagAngDiff();

		metaData = angDiff + magDiff + magAng + scaleDoLP;

		int rows = angDiff.rows;
		int cols = angDiff.cols;

		//add low confidence of areas with higher MagAngDiff and dolp by factor of 10 and 20 respectively
		//note: the higher the value of meta data the lower the confidence
		for (int r = 0; r < rows; ++r) {
			double *PmetaData = metaData.ptr<double>(r);
			double *PmagAng = magAng.ptr<double>(r);
			double *PscaleDoLP = scaleDoLP.ptr<double>(r);
			for (int c = 0; c < cols; ++c) {
				if (*PscaleDoLP > 0.25)*PmetaData *= 20.0;
				if (*PmagAng > 0.75)*PmetaData *= 10.0;
				++PmetaData;
				++PscaleDoLP;
				++PmagAng;
			}
		}

	}

	void pcdFilterPolarization::TrimPointCloud(polarization_Parameters& pol, Transformation& Trans) {

		//test for the validity of the input 
		if (morphOutput.empty())error("Error: morphOutput is not valid: ", "TrimPointCloud()");

		//convert to 8-bit single channel 
		morphOutput.convertTo(morphOutput, CV_8U);

		//invert the gray level pixel using threshold function
		cv::threshold(morphOutput, morphOutput, 0.0, 1.0, cv::ThresholdTypes::THRESH_BINARY_INV);

		//remove the rgb color affected by multireflection
		cv::multiply(Trans.myParameters()->r, morphOutput, Trans.myParameters()->r);
		cv::multiply(Trans.myParameters()->g, morphOutput, Trans.myParameters()->g);
		cv::multiply(Trans.myParameters()->b, morphOutput, Trans.myParameters()->b);

		//do the same to  the 3D points data set
		morphOutput.convertTo(morphOutput, Trans.myParameters()->y.type());

		cv::multiply(Trans.myParameters()->x, morphOutput, Trans.myParameters()->x);
		cv::multiply(Trans.myParameters()->y, morphOutput, Trans.myParameters()->y);
		cv::multiply(Trans.myParameters()->z, morphOutput, Trans.myParameters()->z);

		//Also do thesame to the computed meta data information
		morphOutput.convertTo(morphOutput, angDiff.type());

		cv::multiply(pol.myParameters()->DoLP, morphOutput, pol.myParameters()->DoLP);
		cv::multiply(angDiff, morphOutput, angDiff);
		cv::multiply(magDiff, morphOutput, magDiff);
		cv::multiply(magAng, morphOutput, magAng);
	}

	void pcdFilterPolarization::SaveMetaData(const Transformation& Trans) {
		int cont = 1;
		string fil = v_pcdOutputFileName + ".txt";
		ofstream myfile(fil, std::ios::trunc);

		if (myfile) {
			myfile << "X" << "\t" << "Y" << "\t" << "Z" << "\t" << "R" << "\t" << "G" << "\t" << "B"
				<< "\t" << "AngDiff" << "\t" << "MagDiff" << "\t" << "AngMagDiff" << "\t" << "Result\n";

			for (int r = 0; r < magAng.rows; ++r) {
				float *Px = Trans.myParameters()->x.ptr<float>(r);
				float *Py = Trans.myParameters()->y.ptr<float>(r);
				float *Pz = Trans.myParameters()->z.ptr<float>(r);

				uchar *Pr = Trans.myParameters()->r.ptr<uchar>(r);
				uchar *Pg = Trans.myParameters()->g.ptr<uchar>(r);
				uchar *Pb = Trans.myParameters()->b.ptr<uchar>(r);

				/*float *Pnx = Transformation::myParameters()->rotNx.ptr<float>(r);
				float *Pny = Transformation::myParameters()->rotNy.ptr<float>(r);
				float *Pnz = Transformation::myParameters()->rotNz.ptr<float>(r);

				double *Ppnx = polarization_Parameters::myParameters()->Normal_x.ptr<double>(r);
				double *Ppny = polarization_Parameters::myParameters()->Normal_y.ptr<double>(r);
				double *Ppnz = polarization_Parameters::myParameters()->Normal_z.ptr<double>(r);*/

				double *PmorphOutput = morphOutput.ptr<double>(r);

				double *PangD = angDiff.ptr<double>(r);
				double *PmagD = magDiff.ptr<double>(r);
				double *PmagAng = magAng.ptr<double>(r);

				for (int c = 0; c < magAng.cols; ++c) {
					if ((*Px != 0.f || *Py != 0.f || *Pz != 0.f)) {
						myfile
							//<< cont << "\t"
							<< *Px << "\t" << *Py << "\t" << *Pz << "\t"
							<< (int)*Pr << "\t" << (int)*Pg << "\t" << (int)*Pb << "\t"
							/*			<< *Ppnx << "\t" << *Ppny << "\t" << *Ppnz << "\t"
										<< *Pnx << "\t" << *Pny << "\t" << *Pnz << "\t"*/
							<< *PangD << "\t" << *PmagD << "\t" << *PmagAng << "\t"
							<< *PmorphOutput
							<< endl;
					}
					++Px; ++Py; ++Pz;
					++Pr; ++Pg; ++Pb;
					/*++Pnx; ++Pny; ++Pnz;
					++Ppnx; ++Ppny; ++Ppnz;*/
					++PangD; ++PmagD; ++PmagAng; ++PmorphOutput;
					//++cont;
				}
			}

		}
		else error("unable to open given output file info file");
	}

	void pcdFilterPolarization::setUp(const string &filename) {

			// Create empty property tree object
			pt::ptree tree;

			// Parse the XML into the property tree.
			pt::read_xml(filename, tree);

			// Use get_child to find the node containing the filtering settings, and iterate over
			// its children. If the path cannot be resolved, get_child throws.
			const string headerName = "polarization_settings";
			BOOST_FOREACH(pt::ptree::value_type &v, tree.get_child(headerName)) {
				// The data function is used to access the data stored in a node.

				if (v.first == "polarization_core")//check the settings for the polarization information
				{
					BOOST_FOREACH(pt::ptree::value_type &p, tree.get_child(headerName + "." + v.first))
					{
						if (p.first == "ImgFileName")v_imgFileName = p.second.data();
						else if (p.first == "Refractive_index")stringstream(p.second.data()) >> v_refind;
						else if (p.first == "Surface_type")stringstream(p.second.data()) >> v_surface_type;
					}

				}
				else if (v.first == "demosaic")//check the settings for the demosaicing
				{
					BOOST_FOREACH(pt::ptree::value_type &d, tree.get_child(headerName + "." + v.first))
					{
						if (d.first == "Demosaic_method")stringstream(d.second.data()) >> v_demosaic_method;
						else if (d.first == "OrientPolAngleR0C0")stringstream(d.second.data()) >> v_orientPolAngleR0C0;
						else if (d.first == "OrientPolAngleR1C0")stringstream(d.second.data()) >> v_orientPolAngleR1C0;
						else if (d.first == "OrientPolAngleR0C1")stringstream(d.second.data()) >> v_orientPolAngleR0C1;
						else if (d.first == "OrientPolAngleR1C1")stringstream(d.second.data()) >> v_orientPolAngleR1C1;
					}
				}
				else if (v.first == "transformation")//check the settings for the transformation
				{
					BOOST_FOREACH(pt::ptree::value_type &t, tree.get_child(headerName + "." + v.first))
					{
						if (t.first == "ImgPointFileName")v_imgPointFileName = t.second.data();
						else if (t.first == "ObjPointFileName")v_objPointFileName = t.second.data();
						else if (t.first == "PcdFileName")v_pcdFileName = t.second.data();
						else if (t.first == "SetRadiusSize")stringstream(t.second.data()) >> v_radius_size;
						else if (t.first == "ComputeNormal")stringstream(t.second.data()) >> v_calc_norm;
					}
				}
				else if (v.first == "filtering")//check the settings for the filtering
				{
					BOOST_FOREACH(pt::ptree::value_type &f, tree.get_child(headerName + "." + v.first))
					{
						if (f.first == "StdErrMeanVal")stringstream(f.second.data()) >> v_stdErrMeanVal;
						else if (f.first == "StdErrMeanKernSize")stringstream(f.second.data()) >> v_stdErrMeanKernSize;
						else if (f.first == "MorphKernSize")stringstream(f.second.data()) >> v_morphKernSize;
						else if (f.first == "CalibrateCam")stringstream(f.second.data()) >> v_calibrateCam;
						else if (f.first == "OutputfileName")stringstream(f.second.data()) >> v_pcdOutputFileName;
					}
				}
				else if (v.first == "camera_calibration")//check the settings for only output filename for the calibration
				{
					BOOST_FOREACH(pt::ptree::value_type &c, tree.get_child(headerName + "." + v.first))
					{
						if (c.first == "Write_outputFileName")
						{
							v_outputFileName = c.second.data();
							break;
						}
					}
				}
			}
			//validate the settings file to ensure that all is well
			validateSettings();
	}

	void pcdFilterPolarization::validateSettings() {
		//check for the polarization image file 
		ifstream ist(v_imgFileName.c_str());
		if (!ist)error("Error: unable to open polarization image file");
		ist.clear();
		ist.close();

		//check for refractive index
		if (v_refind <= 0)error("Error: Invalid refractive index");

		//check for surface type: diffuse or specular
		if (!(v_surface_type == int(polarization_Parameters::DIFFUSE) || v_surface_type == polarization_Parameters::SPECULAR))
			error("Error: Invalid surface type: diffuse or specular?");

		//check for validity of the orientation of polarizer filter angles overlaid on the camera
		vector<int>ori;
		ori.push_back(v_orientPolAngleR0C0);
		ori.push_back(v_orientPolAngleR1C0);
		ori.push_back(v_orientPolAngleR0C1);
		ori.push_back(v_orientPolAngleR1C1);
		sort(ori.begin(), ori.end());
		for (int i = 1; i < ori.size(); ++i) {
			if (ori.at(i) == ori.at(i - 1))
				error("Error: the orientation angle is duplicated: ", ori.at(i));
			if (!(ori.at(i) == 0 || ori.at(i) == 45 || ori.at(i) == 90 || ori.at(i) == 135))
				error("Error: invalid orientation angle: ", ori.at(i));
		}

		//check for demosaic method
		if (!(v_demosaic_method == Demosaic::ADAPTIVE
			|| v_demosaic_method == Demosaic::BILINEAR
			|| v_demosaic_method == Demosaic::BTES
			|| v_demosaic_method == Demosaic::DOWNSAMPLE))
			error("Error: Invalid domosaic method");
		if (v_demosaic_method == Demosaic::BTES)
			error("Sorry: BTES method is not yet implemented in the demosaic algorithm");

		//check for image point file
		ist.open(v_imgPointFileName.c_str());
		if (!ist)error("Error: unable to open image point file");
		ist.clear();
		ist.close();

		//check for object point file
		ist.open(v_objPointFileName.c_str());
		if (!ist)error("Error: unable to open object point file");
		ist.clear();
		ist.close();

		//check for point cloud file
		ist.open(v_pcdFileName.c_str());
		if (!ist)error("Error: unable to open point cloud file");
		ist.clear();
		ist.close();

		//check for the validity of the radius size for normal vector computation
		if (v_radius_size <= 0)
			error("Error: invalid value of the radius search size for normal vector computation: ", v_radius_size);

		//check for the validity of normal vector computation
		if (!(v_calc_norm == 1 || v_calc_norm == 0))
			error("Error: To compute normal vector ? set the variable either true(1) or false(0) but not: ", v_calc_norm);

		//check for the validity for calibrating camera
		if (!(v_calibrateCam == 1 || v_calibrateCam == 0))
			error("Error: To calibrate camera? set the variable either true(1) or false(0) but not: ", v_calibrateCam);

		//check for the validity of the standard error of the mean for filtering
		if (v_stdErrMeanVal <= 0)
			error("Error: invalid value for the standard error of the mean: ", v_stdErrMeanVal);

		//check for the validity of the standard error of the mean kernel size for filtering
		if (v_stdErrMeanKernSize <= 0)
			error("Error: invalid value for the standard error of the mean kernel size: ", v_stdErrMeanKernSize);


		//check for the validity of the morphology kernel size for filtering
		if (v_morphKernSize <= 0)
			error("Error: invalid value for the morphology kernel size: ", v_morphKernSize);
	}
}