#include "Demosaic.h"

namespace Demo {
	Demosaic::Demosaic()
	{
		typ = "";
	}

	Demosaic::~Demosaic()
	{
	}

	void Demosaic::setMethod(intTypes typ) {
		switch (typ)
		{
		case Demo::Demosaic::ADAPTIVE:
			this->typ = "adaptive";
			break;
		case Demo::Demosaic::BILINEAR:
			this->typ = "bilinear";
			break;
		default:
			break;
		}
	}
	int Demosaic::exeDemosaic() {
		this->Initialize();

		if (this->myParameters.Input.size() != 1) {
			cerr << "there is no valid input size" << endl;
			return -1;
		}

		if (this->typ == "adaptive")interpolateAdaptive(this->myParameters.Input, this->myParameters.Output);
		else if (this->typ == "bilinear")interpolateBilinear(this->myParameters.Input, this->myParameters.Output);
		else {
			cerr << "there is no valid method type: please check the implemented methods" << endl;
			return -1;
		}

		this->myParameters.Output[0].convertTo(this->myParameters.Output[0], CV_8U);
		this->myParameters.Output[1].convertTo(this->myParameters.Output[1], CV_8U);
		this->myParameters.Output[2].convertTo(this->myParameters.Output[2], CV_8U);
		this->myParameters.Output[3].convertTo(this->myParameters.Output[3], CV_8U);

		imwrite("Angle0.png", this->myParameters.Output.at(0));
		imwrite("Angle45.png", this->myParameters.Output.at(1));
		imwrite("Angle90.png", this->myParameters.Output.at(2));
		imwrite("Angle135.png", this->myParameters.Output.at(3));
		return 0;
	}


	int  Demosaic::interpolateAdaptive(std::vector<cv::Mat>input, std::vector<cv::Mat>&output) {

		int rows = input[0].rows;
		int cols = input[0].cols;
		Mat im0, im45, im135, im90;

		//split the image to calculate imtensity
		imgSplit(input.at(0), output);


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

		input.push_back(im0);
		input.push_back(im45);
		input.push_back(im90);
		input.push_back(im135);

		S0.convertTo(S0, CV_64FC1);
		input[0].convertTo(input[0], CV_64FC1);

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
		Mat img_pad(rows + 2 * pad, cols + 2 * pad, input[0].type(), Scalar::all(0));
		Mat int_pad(rows + 2 * pad, cols + 2 * pad, input[0].type(), Scalar::all(0));
		Mat ori_pad(rows + 2 * pad, cols + 2 * pad, orient.type(), Scalar::all(0));

		input[0].copyTo(img_pad(Rect(pad, pad, cols, rows)));
		cv::resize(S0, S0, input[0].size(), 0.0, 0.0, CV_INTER_CUBIC);
		S0.copyTo(int_pad(Rect(pad, pad, cols, rows)));
		orient.copyTo(ori_pad(Rect(pad, pad, cols, rows)));

		rows = img_pad.rows;
		cols = img_pad.cols;

		for (int r = pad; r < rows - pad; ++r) {
			for (int c = pad; c < cols - pad; ++c) {

				try {


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
							input[orVal.at<double>(0, 0)].at<double>(r - 1, c - 1) = d;
							input[orVal.at<double>(2, 2)].at<double>(r - 1, c - 1) = d;
							input[orVal.at<double>(0, 2)].at<double>(r - 1, c - 1) = d;
							input[orVal.at<double>(2, 0)].at<double>(r - 1, c - 1) = d;

							//for vertical
							input[orVal.at<double>(0, 1)].at<double>(r - 1, c - 1) = v;
							input[orVal.at<double>(2, 1)].at<double>(r - 1, c - 1) = v;

							//for horizontal
							input[orVal.at<double>(1, 0)].at<double>(r - 1, c - 1) = h;
							input[orVal.at<double>(1, 2)].at<double>(r - 1, c - 1) = h;

							//for center
							input[orVal.at<double>(1, 1)].at<double>(r - 1, c - 1) = dRaw.at<double>(1, 1);
							break;
						}
					}

				}
				catch (exception e) {
					cout << r << endl;
					cout << c << endl;
				}

			}
		}

		output.clear();
		output.push_back(input.at(0).clone());
		output.push_back(input.at(1).clone());
		output.push_back(input.at(2).clone());
		output.push_back(input.at(3).clone());

		return 0;
	}

	int Demosaic::interpolateBilinear(std::vector<cv::Mat>input, std::vector<cv::Mat>&output) {
		Mat im0, im45, im90, im135;

		//separate the image to diffent angles
		BtesSeparate(input.at(0), input);

		BIInterpol(input.at(0), im0);
		BIInterpol(input.at(1), im45);
		BIInterpol(input.at(2), im90);
		BIInterpol(input.at(3), im135);

		output.clear();
		output.push_back(im0);
		output.push_back(im45);
		output.push_back(im90);
		output.push_back(im135);
		return 0;
	}

	void  Demosaic::imgSplit(const Mat input, std::vector<cv::Mat>&output) {
		Mat im0, im45, im90, im135;
		int rows = input.rows / 2;
		int cols = input.cols / 2;

		im0.release();
		im0.create(rows, cols, input.type());
		im45 = im0.clone();
		im90 = im0.clone();
		im135 = im0.clone();

			for (int r = 0; r < rows; ++r) {
				uchar *Pim0 = im0.ptr<uchar>(r);
				uchar *Pim45 = im45.ptr<uchar>(r);
				uchar *Pim90 = im90.ptr<uchar>(r);
				uchar *Pim135 = im135.ptr<uchar>(r);
				for (int c = 0; c < cols; ++c) {
					//create mask
					cv::Mat mask(2, 2, input.type());
					cv::Rect rec(c * 2, r * 2, 2, 2);

					//copy to mask
					input(rec).copyTo(mask);

					//copy to image
					*Pim0 = mask.at<uchar>(0, 1);
					*Pim45 = mask.at<uchar>(0, 0);
					*Pim90 = mask.at<uchar>(1, 0);
					*Pim135 = mask.at<uchar>(1, 1);

					++Pim0;
					++Pim45;
					++Pim90;
					++Pim135;


				}
			}
		
		output.clear();
		output.push_back(im0);
		output.push_back(im45);
		output.push_back(im90);
		output.push_back(im135);
	}


	void Demosaic::BIInterpol(const Mat input, Mat &output) {

		int rows = input.rows;
		int cols = input.cols;
		output.create(rows, cols, CV_8U);
		Mat mask(rows, cols, CV_8UC1, Scalar::all(1));

		Mat kernel1B;
		char kernel4[3 * 3] = { 1, 1,1,
			1, 1,1,
			1,1,1 };
		kernel1B = cv::Mat(3, 3, CV_8U, kernel4).clone();

		//pad the image
		int border = kernel1B.rows / 2;

		Mat Interpolated(rows + 2 * border, cols + 2 * border, input.type(), Scalar::all(0));

		Rect ROI(border, border, cols, rows);

		input.copyTo(Interpolated(ROI));

		int  sum1, sum2, sum3, sumall;

		rows = Interpolated.rows;
		cols = Interpolated.cols;

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

				sumall = (sum1 + sum2 + sum3) / 4;

				*poutput = sumall;

			}

		}
		rows = input.rows;
		cols = input.cols;
		for (int a = 0; a < rows; a++)
		{

			for (int b = 0; b < cols; b++)
			{
				if (input.at<uchar>(a, b) != 0)output.at<uchar>(a, b) = input.at<uchar>(a, b);
			}
		}

	}



	void Demosaic::BtesSeparate(const Mat input, std::vector<cv::Mat>&output) {
		Mat im0, im45, im90, im135;
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

				uchar *pKernel10 = Kernel0.ptr<uchar>(0);
				uchar *pKernel20 = Kernel0.ptr<uchar>(1);
				uchar *pKernel145 = Kernel45.ptr<uchar>(0);
				uchar *pKernel245 = Kernel45.ptr<uchar>(1);
				uchar *pKernel190 = Kernel90.ptr<uchar>(0);
				uchar *pKernel290 = Kernel90.ptr<uchar>(1);
				uchar *pKernel1135 = Kernel135.ptr<uchar>(0);
				uchar *pKernel2135 = Kernel135.ptr<uchar>(1);

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
		output.push_back(im0);
		output.push_back(im45);
		output.push_back(im90);
		output.push_back(im135);
	}

	int Demosaic::Initialize() {
		// --- second Angle ----------------------------------------
		uchar filter0[2 * 2] = { 0, 1,
			0, 0 };


		// --- first Angle ----------------------------------------
		uchar filter45[2 * 2] = { 1, 0,
			0, 0 };


		// ---third Angle----------------------------------------
		uchar filter90[2 * 2] = { 0, 0,
			1, 0 };

		// --- forth Angle----------------------------------------
		uchar filter135[2 * 2] = { 0, 0,
			0, 1 };

		Kernel0 = Mat(2, 2, CV_8U, filter0).clone();
		Kernel45 = Mat(2, 2, CV_8U, filter45).clone();
		Kernel90 = Mat(2, 2, CV_8U, filter90).clone();
		Kernel135 = Mat(2, 2, CV_8U, filter135).clone();
		return 0;
	}

}