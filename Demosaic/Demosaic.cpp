#include "Demosaic.h"

namespace POL {

	Demosaic::Demosaic(int typ)
	{
		m_parameters = new ContainerDem::Parameters();
		m_typ = typ;
	}

	Demosaic::~Demosaic()
	{
		delete m_parameters;
	}
	

	void Demosaic::setInput(int argv, char **argc) {
		//check if the input arguement is passed
		//only single and raw polarization image file is allowed for demosaicing
		if (argv != 2) {
			error("Invalid input detected\n");
			
		}

		//declare variable for the image buffer and read image from file
		Mat data = imread(argc[1]);

		//convert image to gray
		cvtColor(data, myParameters()->Input, CV_BGR2GRAY);

	}
	void Demosaic::exeDemosaic() {
		Initialize();

		if (myParameters()->Input.empty()) {
			error("there is no valid input for the function: ", "exeDemosaic()");
		}

		if (m_typ == Demosaic::ADAPTIVE)interpolateAdaptive(myParameters()->Input, myParameters()->Output);
		else if (m_typ == Demosaic::BILINEAR)interpolateBilinear(myParameters()->Input, myParameters()->Output);
		else if (m_typ == Demosaic::DOWNSAMPLE) {
			imgSplit(myParameters()->Input, myParameters()->Output);
			map<int, cv::Mat>mymap;

			mymap[myParameters()->orientPolAngleR0C0] = myParameters()->Output[0].clone();
			mymap[myParameters()->orientPolAngleR1C0] = myParameters()->Output[1].clone();
			mymap[myParameters()->orientPolAngleR0C1] = myParameters()->Output[2].clone();
			mymap[myParameters()->orientPolAngleR1C1] = myParameters()->Output[3].clone();

			myParameters()->Output.clear();
			myParameters()->Output.push_back(mymap.at(0).clone());
			myParameters()->Output.push_back(mymap.at(45).clone());
			myParameters()->Output.push_back(mymap.at(90).clone());
			myParameters()->Output.push_back(mymap.at(135).clone());
		}
		else {
			error("there is no valid method m_type for demosaicing: please check the implemented methods");
		}

		if (myParameters()->Output.empty())
			error("invalid output image buffer for demosaic container: ","exeDemosaic()");

		myParameters()->Output[0].convertTo(myParameters()->Output[0], CV_8U);
		myParameters()->Output[1].convertTo(myParameters()->Output[1], CV_8U);
		myParameters()->Output[2].convertTo(myParameters()->Output[2], CV_8U);
		myParameters()->Output[3].convertTo(myParameters()->Output[3], CV_8U);
	}


	void  Demosaic::interpolateAdaptive(const cv::Mat &input, std::vector<cv::Mat>&output) {
		if (input.empty())error("invalid input image with the function call: ", "interpolateAdaptive()");
		int rows = input.rows;
		int cols = input.cols;
		Mat im0, im45, im135, im90;

		//split the image to calculate imtensity
		imgSplit(input, output);


		output.at(0).convertTo(output.at(0), CV_64FC1);
		output.at(1).convertTo(output.at(1), CV_64FC1);
		output.at(2).convertTo(output.at(2), CV_64FC1);
		output.at(3).convertTo(output.at(3), CV_64FC1);

		cv::Mat S0 = 0.25 * (output.at(0) + output.at(1) + output.at(2) + output.at(3));

		im0.create(rows, cols, CV_64FC1);
		im0 = Scalar::all(0);
		im45 = im0.clone();
		im90 = im0.clone();
		im135 = im0.clone();

		std::vector<cv::Mat>tmp;
		tmp.push_back(input.clone());
		tmp.push_back(im0);
		tmp.push_back(im45);
		tmp.push_back(im90);
		tmp.push_back(im135);

		S0.convertTo(S0, CV_64FC1);
		tmp[0].convertTo(tmp[0], CV_64FC1);

		//defined the kernels
		double kernelv[9] = {
								0,1, 0,
								0, 0, 0,
								0, 1, 0 };

		double kernelh[9] = {
								0,0, 0,
								1, 0, 1,
								0, 0, 0 };

		double kerneld[9] = {
								1,0, 1,
								0, 0, 0,
								1, 0, 1 };


		Mat maskd = Mat(3, 3, CV_64FC1, kerneld).clone();
		Mat maskv = Mat(3, 3, CV_64FC1, kernelv).clone();
		Mat maskh = Mat(3, 3, CV_64FC1, kernelh).clone();

		//get an image with the orientation of the polarization fllter
		Mat orient(rows, cols, CV_64FC1);
		for (int r = 0; r < rows; r += 2) {
			for (int c = 0; c < cols; c += 2) {
				//create mask
				orient.at<double>(r, c) = 2;// 45;
				orient.at<double>(r + 1, c + 1) = 4;// 135;
				orient.at<double>(r, c + 1) = 1;// 0;
				orient.at<double>(r + 1, c) = 3;// 90;
			}
		}

		//calc the missing pixels
		//pad images
		int pad = 1;
		Mat img_pad(rows + 2 * pad, cols + 2 * pad, tmp[0].type(), Scalar::all(0));
		Mat int_pad(rows + 2 * pad, cols + 2 * pad, tmp[0].type(), Scalar::all(0));
		Mat ori_pad(rows + 2 * pad, cols + 2 * pad, orient.type(), Scalar::all(0));

		tmp[0].copyTo(img_pad(Rect(pad, pad, cols, rows)));
		cv::resize(S0, S0, tmp[0].size(), 0.0, 0.0, CV_INTER_CUBIC);
		S0.copyTo(int_pad(Rect(pad, pad, cols, rows)));
		orient.copyTo(ori_pad(Rect(pad, pad, cols, rows)));

		rows = img_pad.rows;
		cols = img_pad.cols;

		for (int r = pad; r < rows - pad; ++r) {
			for (int c = pad; c < cols - pad; ++c) {


					//create mask
					Mat dRaw(3, 3, img_pad.type(), Scalar::all(0));
					Mat intVal = dRaw.clone();
					Mat orVal(3, 3, orient.type(), Scalar::all(0));

					img_pad(Rect(c - pad, r - pad, 3, 3)).copyTo(dRaw);
					int_pad(Rect(c - pad, r - pad, 3, 3)).copyTo(intVal);
					ori_pad(Rect(c - pad, r - pad, 3, 3)).copyTo(orVal);

					//get the intensity similarity
					Mat Intsim(intVal.size(), intVal.type());
					vector<double>intstd, mnVal;
					meanStdDev(intVal, mnVal, intstd);
					for (int m = 0; m < 3; ++m) {
						for (int n = 0; n < 3; ++n) {
							Intsim.at<double>(m, n) = (1 / (std::sqrt(2 * CV_PI*intstd[0]))) *
								std::exp(-(pow((intVal.at<double>(1, 1) - intVal.at<double>(m, n)), 2.0)) / (2 * intstd[0] * intstd[0]));
						}
					}

					Mat bv, bh, bd;
					Mat D, V, H;

					//vertivcal
					cv::multiply(Intsim, maskv, bv);
					Scalar resv = cv::sum(bv);
					bv = bv / resv[0];
					cv::multiply(dRaw, maskv, V);
					cv::multiply(V, bv, V);
					resv = cv::sum(V);

					//horizontal
					cv::multiply(Intsim, maskh, bh);
					Scalar resh = cv::sum(bh);
					bh = bh / resh[0];
					cv::multiply(dRaw, maskh, H);
					cv::multiply(H, bh, H);
					resh = cv::sum(H);
					//diagonal
					cv::multiply(Intsim, maskd, bd);
					Scalar resd = cv::sum(bd);
					bd = bd / resd[0];
					cv::multiply(dRaw, maskd, D);
					cv::multiply(D, bd, D);
					resd = cv::sum(D);

					//reconst
					double d = resv[0] + resh[0] - dRaw.at<double>(1, 1);
					double v = dRaw.at<double>(1, 1) + resd[0] - resh[0];
					double h = dRaw.at<double>(1, 1) + resd[0] - resv[0];


					for (int k = 0; k < 9; ++k) {
						if (orVal.at<double>(k) == 1 || orVal.at<double>(k) == 2 || orVal.at<double>(k) == 3 || orVal.at<double>(k) == 4) {
							//for diagonal
							//cout << orVal<< endl;
							if (tmp.size() != 5)error("range error at vector tmp with function call: ", "interpolateAdaptive()");
							tmp[orVal.at<double>(0, 0)].at<double>(r - 1, c - 1) = d;
							tmp[orVal.at<double>(2, 2)].at<double>(r - 1, c - 1) = d;
							tmp[orVal.at<double>(0, 2)].at<double>(r - 1, c - 1) = d;
							tmp[orVal.at<double>(2, 0)].at<double>(r - 1, c - 1) = d;

							//for vertical
							tmp[orVal.at<double>(0, 1)].at<double>(r - 1, c - 1) = v;
							tmp[orVal.at<double>(2, 1)].at<double>(r - 1, c - 1) = v;

							//for horizontal
							tmp[orVal.at<double>(1, 0)].at<double>(r - 1, c - 1) = h;
							tmp[orVal.at<double>(1, 2)].at<double>(r - 1, c - 1) = h;

							//for center
							tmp[orVal.at<double>(1, 1)].at<double>(r - 1, c - 1) = dRaw.at<double>(1, 1);
							break;
						}
					}

		

			}
		}

		output.clear();
		if (tmp.at(1).empty())error("invalid output image at tmp.at(1) with the function call: ", "interpolateAdaptive()");
		if (tmp.at(2).empty())error("invalid output image at tmp.at(2) with the function call: ", "interpolateAdaptive()");
		if (tmp.at(3).empty())error("invalid output image at tmp.at(3) with the function call: ", "interpolateAdaptive()");
		if (tmp.at(4).empty())error("invalid output image at tmp.at(4) with the function call: ", "interpolateAdaptive()");
		output.push_back(tmp.at(1).clone());
		output.push_back(tmp.at(2).clone());
		output.push_back(tmp.at(3).clone());
		output.push_back(tmp.at(4).clone());
	}

	void Demosaic::interpolateBilinear(const cv::Mat &input, std::vector<cv::Mat>&output) {
		Mat im0, im45, im90, im135;
		if (input.empty())error("invalid input image with the function call: ", "interpolateBilinear()");
		//separate the image to diffent angles
		std::vector<cv::Mat> output_tmp(4);
		BtesSeparate(input, output_tmp);

		//apply bilinear interpolation for each of the separated image
		BIInterpol(output_tmp.at(0), im0);
		BIInterpol(output_tmp.at(1), im45);
		BIInterpol(output_tmp.at(2), im90);
		BIInterpol(output_tmp.at(3), im135);

		output.clear();
		if (im0.empty())error("invalid output image at im0 with the function call: ", "interpolateBilinear()");
		if (im45.empty())error("invalid output image at im45 with the function call: ", "interpolateBilinear()");
		if (im135.empty())error("invalid output image at im135 with the function call: ", "interpolateBilinear()");
		if (im90.empty())error("invalid output image at im90 with the function call: ", "interpolateBilinear()");
		output.push_back(im0);
		output.push_back(im45);
		output.push_back(im90);
		output.push_back(im135);
	}

	void Demosaic::BIInterpol(const Mat &input, Mat &output) {
		if (input.empty())error("invalid input image with the function call: ", "BIInterpol()");
		int rows = input.rows;
		int cols = input.cols;
		output.create(rows, cols, CV_8U);

		//create the kernel for bilinear interpolation

		Mat kernel1B;
		char kernel[3 * 3] = { 1, 1,1,
			1, 1,1,
			1,1,1 };
		kernel1B = cv::Mat(3, 3, CV_8U, kernel).clone();

		//pad the image
		int border = kernel1B.rows / 2;

		Mat Interpolated(rows + 2 * border, cols + 2 * border, input.type(), Scalar::all(0));

		Rect ROI(border, border, cols, rows);

		input.copyTo(Interpolated(ROI));

		int  sum1, sum2, sum3, sumall;

		rows = Interpolated.rows;
		cols = Interpolated.cols;
		//loop through the image file using the //3-by-3 kernel and... 
		//apply bilinear interpolation to recover the missing polarizer angle per pixel
		for (int a = 0; a < (rows - 2 * border); a++)
		{

			for (int b = 0; b < (cols - 2 * border); b++)
			{
				const uchar	*pInput1 = Interpolated.ptr<uchar>(a) + b;
				const uchar *pInput2 = Interpolated.ptr<uchar>(a + 1) + b;
				const uchar *pInput4 = Interpolated.ptr<uchar>(a + 2) + b;


				const uchar *pKernel1 = kernel1B.ptr<uchar>(0);
				const uchar *pKernel2 = kernel1B.ptr<uchar>(1);
				const uchar *pKernel3 = kernel1B.ptr<uchar>(2);

				uchar *poutput = output.ptr<uchar>(a) + b;
				sum1 = 0.0f;
				sum2 = 0.0f;
				sum3 = 0.0f;
				for (int k = 0; k < 3; k++) {
					sum1 += (*pInput1) * (*pKernel1);
					sum2 += (*pInput2)* (*pKernel2);
					sum3 += (*pInput4) * (*pKernel3);

					++pInput1;
					++pInput2;
					++pInput4;

					++pKernel1;
					++pKernel2;
					++pKernel3;

				}

				sumall = (sum1 + sum2 + sum3) / 4; // we divide by 4 
				// this is because at each 3-by-3 pixel size only 4 pixel are valid out of 9
				//except towards the image border off course but the effect is negligible

				*poutput = sumall;

			}

		}
		rows = input.rows;
		cols = input.cols;
		//after we return the value of the original valid input to reduce the effect of interpolation artifacts
		for (int a = 0; a < rows; a++)
		{

			for (int b = 0; b < cols; b++)
			{
				if (input.at<uchar>(a, b) != 0)output.at<uchar>(a, b) = input.at<uchar>(a, b);
			}
		}
		if (output.empty())error("invalid output image with the function call: ", "BIInterpol()");
	}



	void Demosaic::BtesSeparate(const Mat &input, std::vector<cv::Mat>&output)
		//this function separate the value of the raw polarization image file into 4 different angles of the polarizer
			//this function retain the size of the image after separation per polarizer angle
	{
		if (input.empty())error("invalid input image with the function call: ", "BtesSeparate()");
		Mat im0, im45, im90, im135; //this is image at different polarizer angle
		int rows = input.rows;
		int cols = input.cols;
		im0.create(rows, cols, CV_8U);
		im90 = im0.clone();
		im45 = im0.clone();
		im135 = im0.clone();

		for (int r = 0; r < (rows); r += 2)
		{

			for (int c = 0; c < cols; c += 2)
			{
				const uchar *pInputAbove = input.ptr<uchar>(r) + c;
				const uchar *pInput = input.ptr<uchar>(r + 1) + c;

				uchar *pOutput10 = im0.ptr<uchar>(r) + c;
				uchar *pOutput20 = im0.ptr<uchar>(r + 1) + c;
				uchar *pOutput145 = im45.ptr<uchar>(r) + c;
				uchar *pOutput245 = im45.ptr<uchar>(r + 1) + c;
				uchar *pOutput190 = im90.ptr<uchar>(r) + c;
				uchar *pOutput290 = im90.ptr<uchar>(r + 1) + c;
				uchar *pOutput1135 = im135.ptr<uchar>(r) + c;
				uchar *pOutput2135 = im135.ptr<uchar>(r + 1) + c;

				uchar *pKernel10 = orientPolAngle.at(0).ptr<uchar>(0);
				uchar *pKernel20 = orientPolAngle.at(0).ptr<uchar>(1);
				uchar *pKernel145 = orientPolAngle.at(45).ptr<uchar>(0);
				uchar *pKernel245 = orientPolAngle.at(45).ptr<uchar>(1);
				uchar *pKernel190 = orientPolAngle.at(90).ptr<uchar>(0);
				uchar *pKernel290 = orientPolAngle.at(90).ptr<uchar>(1);
				uchar *pKernel1135 = orientPolAngle.at(135).ptr<uchar>(0);
				uchar *pKernel2135 = orientPolAngle.at(135).ptr<uchar>(1);

				for (int k = 0; k < 2; ++k)
				{
					*pOutput10 = (*pInputAbove) * (*pKernel10);
					*pOutput20 = (*pInput) * (*pKernel20);

					*pOutput145 = (*pInputAbove) * (*pKernel145);
					*pOutput245 = (*pInput) * (*pKernel245);

					*pOutput190 = (*pInputAbove) * (*pKernel190);
					*pOutput290 = (*pInput) * (*pKernel290);

					*pOutput1135 = (*pInputAbove) * (*pKernel1135);
					*pOutput2135 = (*pInput) * (*pKernel2135);

					++pInputAbove;
					++pInput;

					++pKernel10;
					++pKernel20;
					++pOutput10;
					++pOutput20;

					++pKernel145;
					++pKernel245;
					++pOutput145;
					++pOutput245;

					++pKernel190;
					++pKernel290;
					++pOutput190;
					++pOutput290;

					++pKernel1135;
					++pKernel2135;
					++pOutput1135;
					++pOutput2135;
				}


			}

		}

		output.clear();
		if (im0.empty())error("invalid output image at im0 with the function call: ", "BtesSeparate()");
		if (im45.empty())error("invalid output image at im45 with the function call: ", "BtesSeparate()");
		if (im135.empty())error("invalid output image at im135 with the function call: ", "BtesSeparate()");
		if (im90.empty())error("invalid output image at im90 with the function call: ", "BtesSeparate()");
		output.push_back(im0);
		output.push_back(im45);
		output.push_back(im90);
		output.push_back(im135);
	}

	void Demosaic::Initialize() {
		// --- second Angle ----------------------------------------
		uchar filter01[2 * 2] = { 0, 1,
			0, 0 };


		// --- first Angle ----------------------------------------
		uchar filter00[2 * 2] = { 1, 0,
			0, 0 };


		// ---third Angle----------------------------------------
		uchar filter10[2 * 2] = { 0, 0,
			1, 0 };

		// --- forth Angle----------------------------------------
		uchar filter11[2 * 2] = { 0, 0,
			0, 1 };

		Kernel00 = Mat(2, 2, CV_8U, filter00).clone();
		Kernel11 = Mat(2, 2, CV_8U, filter11).clone();
		Kernel10 = Mat(2, 2, CV_8U, filter10).clone();
		Kernel01 = Mat(2, 2, CV_8U, filter01).clone();

		//map the output kernel to the correstponding polarizer orientation based on the parameter settings 
		orientPolAngle[myParameters()->orientPolAngleR0C0] = Kernel00.clone();
		orientPolAngle[myParameters()->orientPolAngleR0C1] = Kernel01.clone();
		orientPolAngle[myParameters()->orientPolAngleR1C0] = Kernel10.clone();
		orientPolAngle[myParameters()->orientPolAngleR1C1] = Kernel11.clone();

	}

}