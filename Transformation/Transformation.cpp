#include <direct.h>
#include<string>
#include<chrono>
#include "Transformation.h"

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include "opencv2/calib3d/calib3d.hpp"

namespace POL {


	void Transformation::loadFile(const string &input, vector<double> &output) {
		ifstream myfile(input.c_str());

		if (myfile) //test if file opening if succedded
		{

			string line;
			vector<string>strV;
			while (!myfile.eof()) {
				getline(myfile, line);
				if (line == "")continue;

				//tokenize the line
				boost::trim(line);
				boost::split(strV, line, boost::is_any_of("\t\r"), boost::token_compress_on);
				if (strV.size() == 2) {
					output.push_back((double)atof(strV[0].c_str()));
					output.push_back((double)atof(strV[1].c_str()));
				}
				else if (strV.size() == 3) {
					output.push_back((double)atof(strV[0].c_str()));
					output.push_back((double)atof(strV[1].c_str()));
					output.push_back((double)atof(strV[2].c_str()));
				}

			}
		}
		else {
			error("error opening file with the funtion call: ", "loadFile()");
		}
		//test the state of the ifstream after loading file
		if(myfile.fail())error("error: failed to read from the input file: ", "loadFile()");
		else if(myfile.bad())error("error: Bad input file path: ", "loadFile()");
	//test for the validity of the output 
		if(output.empty())error("error: output file is not valid: ", "loadFile()");

	}


	void Transformation::loadFilePCD(const string& input, pcl::PointCloud<pcl::PointXYZRGBNormal>&output, bool compNormal)
	{
			//check if normal vector has to be calculated
		if (!compNormal) {
			pcl::io::loadPCDFile(input,output);
		}
		else {
			if (compNormal && radius_search <= 0.f)
				error("Error: radius search size must be greater than zero for normal vector computation");

			//load the point cloud pointer
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cl(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::io::loadPCDFile(input, *cl);
			cout << "size of cloud: " << cl->size() << endl;
			// Create the normal estimation class, and pass the input dataset to it
			pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;

			ne.setInputCloud(cl);

			// Create an empty kdtree representation, and pass it to the normal estimation object.
			// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
			pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
			ne.setSearchMethod(tree);

			// Use all neighbors in a sphere of radius 3cm
			ne.setRadiusSearch(radius_search);

			// Output datasets
			pcl::PointCloud<pcl::Normal>::Ptr clN(new pcl::PointCloud<pcl::Normal>);

			// Compute the features
			ne.compute(*clN);

			//copy cloud to pcd container

			pcl::PointCloud<pcl::Normal>::iterator itN = clN->begin();

			/*	ofstream myf("Text.txt", std::ios::trunc);
				if (myf.is_open())
					{*/
			for (pcl::PointCloud<pcl::PointXYZRGB>::iterator itx = cl->begin(); itx != cl->end(); ++itx, ++itN) {
				pcl::PointXYZRGBNormal tmp;
				tmp.x = itx->x;
				tmp.y = itx->y;
				tmp.z = itx->z;
				tmp.r = itx->r;
				tmp.g = itx->g;
				tmp.b = itx->b;
				tmp.normal_x = isnan(itN->normal_x) ? 0 : itN->normal_x;
				tmp.normal_y = isnan(itN->normal_y) ? 0 : itN->normal_y;
				tmp.normal_z = isnan(itN->normal_z) ? 0 : itN->normal_z;
				output.push_back(tmp);

			}

		}
		//test for the validity of the output 
		if (output.empty())error("error: output file is not valid: ", "loadFilePCD()");
	}



	void Transformation::getRotTrans(const string& imPoints, const string& objPoints, const cv::Mat& camMatrix, const cv::Mat& distCoeffs, cv::Mat &rotm, cv::Mat &tranv)
	{
		vector<std::pair<int, std::vector<double>>>pairvector;
		map<int, vector<double>>mytest;
		//load image points
		loadFile(imPoints.c_str(), KimagePoint);

		//load object points
		loadFile(objPoints.c_str(), KObjectPoint);


		/*cout << "\n********************image points:**********************" << endl;
		for (vector<double>::iterator it = KimagePoint.begin(); it != KimagePoint.end(); it += 2)
		{
			std::cout << *it << " " << *(it + 1) << endl;
		}

		std::cout << "\n********************object points:**********************" << endl;
		for (vector<double>::iterator it = KObjectPoint.begin(); it != KObjectPoint.end(); it += 3)
		{

			std::cout << *it << " " << *(it + 1) << " " << *(it + 2) << endl;
		}*/



		ImagePointMat = cv::Mat(KimagePoint.size() / 2, 2, CV_64FC1, KimagePoint.data()).clone();
		ObjectPointMat = cv::Mat(KObjectPoint.size() / 3, 3, CV_64FC1, KObjectPoint.data()).clone();

		//std::cout << "\nInitial cameraMatrix: " << camMatrix << std::endl;

		cv::solvePnP(ObjectPointMat, ImagePointMat, camMatrix, distCoeffs, rotm, tranv);

	/*	std::cout << "rvec: " << rotm << std::endl;
		std::cout << "tvec: " << tranv << std::endl;
*/
		Rodrigues(rotm, rotm);

		tranv.convertTo(tranv, CV_32FC1);
		rotm.convertTo(rotm, CV_32FC1);

	//	std::cout << "Rodvec: " << rotm << std::endl;

		//test for the validity of the output 
		if (tranv.empty() || rotm.empty())error("error: translation or rotational vector is not valid: ", "getRotTrans()");
	}



	void Transformation::ProjectPoint3D(int argv, char **argc,const cv::Mat &camMatrix,
		const cv::Mat &distCoeff, int imHeight, int imWidth, bool compNorm) {
		if (!(argv == 4)) {//check if the input arguement is passed
			error("Invalid input detected\n");

		}

		if (compNorm && radius_search <= 0.f)
			error("Error: radius search size must be greater than zero for normal vector computation");

		if (camMatrix.empty())error("Error: camera matrix is empty");
		if (distCoeff.empty())error("Error: distortion coefficient is empty");
		if (imHeight <=0 || imWidth <= 0)error("Error: invalid image size");

		getRotTrans(argc[1], argc[2], camMatrix, distCoeff,
			myParameters()->rotv, myParameters()->tranv);

		//load 3D point points and set the
		loadFilePCD(argc[3], myParameters()->original, compNorm);

		//std::cout << "\nfinish loading 3D points... " << endl;

		//project the point to the image
		ProjectPoint3D(myParameters()->original,
			myParameters()->projected, camMatrix,
			myParameters()->rotv, myParameters()->tranv);

		//std::cout << "\nfinishing projecting 3D points: " << endl;

		// filter along z-dimension while looking at the camera viewing direction 
		camViewFilterZdim(imHeight,imWidth);
	}


	void Transformation::ProjectPoint3D(pcl::PointCloud<pcl::PointXYZRGBNormal>& input,
		pcl::PointCloud<pcl::PointXYZRGBNormal>&output, const cv::Mat& camMatrix, const cv::Mat& rotv, const cv::Mat& tranv)
	{
		//test for the validity of the input 
		if (tranv.empty())error("Error: translation vector is not valid: ", "ProjectPoint3D()");
		if (input.empty())error("Error: input is not valid: ", "ProjectPoint3D()");
		if (rotv.empty())error("Error: rotational vector is not valid: ", "ProjectPoint3D()");
		if (camMatrix.empty())error("Error: camMatrix matrix is not valid: ", "ProjectPoint3D()");
		cv::Mat camMat_tmp;
		camMatrix.convertTo(camMat_tmp, CV_32FC1);

		for (pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator it = input.begin(); it != input.end(); ++it) {
			//get the xyz data
			cv::Mat xyz(3, 1, CV_32FC1), Nxyz(3, 1, CV_32FC1);
			xyz.at<float>(0) = it->x;
			xyz.at<float>(1) = it->y;
			xyz.at<float>(2) = it->z;
			Nxyz.at<float>(0) = it->normal_x;// itN->normal_x;
			Nxyz.at<float>(1) = it->normal_y;// itN->normal_y;
			Nxyz.at<float>(2) = it->normal_z;// itN->normal_z;

			//rotate the point cloud and normal vector in thesame orientation w.r.t image plane
			cv::Mat Rxyz = rotv * xyz;
			cv::Mat RNxyz = rotv * Nxyz;

			//Translate only the xyz;
			cv::Mat Txyz = Rxyz + tranv;

			//project points to to image plane
			cv::Mat imgxyz = camMat_tmp * Txyz;

			//normalize point
			float mag = std::sqrt((std::pow(RNxyz.at<float>(0), 2.0) + std::pow(RNxyz.at<float>(1), 2.0) 
				+ std::pow(RNxyz.at<float>(2), 2.0)));
			float nx = 0;
			float ny = 0;
			float nz = 0;
			if (mag != 0) { // only points that mag not zero are considered
				nx = RNxyz.at<float>(0);
				ny = RNxyz.at<float>(1);
				nz = -RNxyz.at<float>(2);
			}


			if (imgxyz.at<float>(2) != 0.f) {
				imgxyz.at<float>(0) = imgxyz.at<float>(0) / imgxyz.at<float>(2);
				imgxyz.at<float>(1) = imgxyz.at<float>(1) / imgxyz.at<float>(2);
			}
			else {
				imgxyz.at<float>(0) = 0.f;
				imgxyz.at<float>(1) = 0.f;
			}

			//return to a new points
			pcl::PointXYZRGBNormal tt;
			//Norm.push_back(pcl::Normal(RNxyz.at<float>(0), RNxyz.at<float>(1), RNxyz.at<float>(2)));


			tt.x = imgxyz.at<float>(0);
			tt.y = imgxyz.at<float>(1);
			tt.z = imgxyz.at<float>(2);
			tt.r = it->r;
			tt.g = it->g;
			tt.b = it->b;
			tt.normal_x = nx;
			tt.normal_y = ny;
			tt.normal_z = nz;

			output.push_back(tt);
		}
		//test for the validity of the output 
		if (output.empty())error("Error: output is not valid: ", "ProjectPoint3D()");
	}

	void Transformation::camViewFilterZdim(int height, int width) {
		cv::Mat Zb, rNx, rNy, rNz;
		cv::Mat ox, oy, oz, r, g, b, nx, ny, nz;
		int rows = height;
		int cols = width;
		Zb.create(rows, cols, CV_32FC1);

		nx = Zb.clone();
		nx = cv::Scalar::all(0.f);
		ny = nx.clone();
		nz = nx.clone();
		ox = nx.clone();
		oy = nx.clone();
		oz = nx.clone();
		r.create(rows, cols, CV_8UC1);
		r = cv::Scalar::all(0);
		g = r.clone();
		b = r.clone();

		rNx = nx.clone();
		rNy = nx.clone();
		rNz = nx.clone();

		Zb = cv::Scalar::all(255.0);

		//cv::Mat Temp(Height, Width, CV_8UC3,Scalar(100,45,150));
		cv::Mat Temp(nx.rows, nx.cols, CV_8UC3, cv::Scalar(255, 0, 255));

		int count = 0;

		//test for the validity of the input 
		if (myParameters()->original.empty())error("Error: original point cloud is not valid: ", "camViewFilterZdim()");
		if (myParameters()->projected.empty())error("Error: projected point cloud is not valid: ", "camViewFilterZdim()");

		pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator orig = myParameters()->original.begin();
		for (pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator itx = myParameters()->projected.begin(); itx != myParameters()->projected.end(); ++itx, ++orig)
		{

			int x = 0, y = 0;
			try {

				//remove out of range point and negative cordinate values 
				if ((itx->x >= 0) && (itx->y >= 0) && (itx->x < cols) && (itx->y < rows)) {
					++count;
					y = (uint16_t)itx->y;
					x = (uint16_t)itx->x;
					float getcurZVal = Zb.at<float>(y, x);
					uchar r_ = (uchar)itx->r;
					uchar g_ = (uchar)itx->g;
					uchar b_ = (uchar)itx->b;

					if (getcurZVal == 255.0) {//first time of considering the position

						Zb.at<float>(y, x) = itx->z;

						ox.at<float>(y, x) = orig->x;
						oy.at<float>(y, x) = orig->y;
						oz.at<float>(y, x) = orig->z;
						r.at<uchar>(y, x) = orig->r;
						g.at<uchar>(y, x) = orig->g;
						b.at<uchar>(y, x) = orig->b;
						nx.at<float>(y, x) = orig->normal_x;
						ny.at<float>(y, x) = orig->normal_y;
						nz.at<float>(y, x) = orig->normal_z;
						rNx.at<float>(y, x) = itx->normal_x;
						rNy.at<float>(y, x) = itx->normal_y;
						rNz.at<float>(y, x) = itx->normal_z;

						Temp.data[((uint16_t)itx->y * Temp.step) + ((uint16_t)itx->x * Temp.elemSize()) + 0] = b_;
						Temp.data[((uint16_t)itx->y * Temp.step) + ((uint16_t)itx->x * Temp.elemSize()) + 1] = g_;
						Temp.data[((uint16_t)itx->y * Temp.step) + ((uint16_t)itx->x * Temp.elemSize()) + 2] = r_;

					}
					else {//the position has been checked before
						float getNewZVal = itx->z;
						if (std::abs(getNewZVal) < std::abs(getcurZVal)) {
							Zb.at<float>(y, x) = getNewZVal;
							ox.at<float>(y, x) = orig->x;
							oy.at<float>(y, x) = orig->y;
							oz.at<float>(y, x) = orig->z;
							r.at<uchar>(y, x) = orig->r;
							g.at<uchar>(y, x) = orig->g;
							b.at<uchar>(y, x) = orig->b;
							nx.at<float>(y, x) = orig->normal_x;
							ny.at<float>(y, x) = orig->normal_y;
							nz.at<float>(y, x) = orig->normal_z;
							rNx.at<float>(y, x) = itx->normal_x;
							rNy.at<float>(y, x) = itx->normal_y;
							rNz.at<float>(y, x) = itx->normal_z;

							Temp.data[((uint16_t)itx->y * Temp.step) + ((uint16_t)itx->x * Temp.elemSize()) + 0] = b_;
							Temp.data[((uint16_t)itx->y * Temp.step) + ((uint16_t)itx->x * Temp.elemSize()) + 1] = g_;
							Temp.data[((uint16_t)itx->y * Temp.step) + ((uint16_t)itx->x * Temp.elemSize()) + 2] = r_;
						}
					}
				}
			}
			catch (...) {
				cout << "x value: " << x << endl;
				cout << "y value: " << y << endl;
			}

		}

		/*cv::imshow("sample", Temp);
		cv::waitKey();*/

		//remove nan
		cv::patchNaNs(rNx, 0.f);
		cv::patchNaNs(rNy, 0.f);
		cv::patchNaNs(rNz, 0.f);

		myParameters()->img = Temp.clone();

		myParameters()->Zbuffer = Zb.clone();

		myParameters()->norm_x = nx.clone();
		myParameters()->norm_y = ny.clone();
		myParameters()->norm_z = nz.clone();

		myParameters()->x = ox.clone();
		myParameters()->y = oy.clone();
		myParameters()->z = oz.clone();

		myParameters()->r = r.clone();
		myParameters()->g = g.clone();
		myParameters()->b = b.clone();

		myParameters()->rotNx = rNx.clone();
		myParameters()->rotNy = rNy.clone();
		myParameters()->rotNz = rNz.clone();
	}



}



