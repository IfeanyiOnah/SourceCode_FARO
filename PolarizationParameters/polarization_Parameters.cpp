#include "polarization_Parameters.h"
#include<map>
namespace POL {

	polarization_Parameters::polarization_Parameters(int typ)
	{
		m_parameters = new ContainerPol::Parameters();
		v_polType = typ;
	}

	polarization_Parameters::~polarization_Parameters()
	{
		delete m_parameters;

	}

	void polarization_Parameters::setInput(int argv, char **argc) {
		if (!(argv == 2 || argv == 5)) {//check if the input arguement is passed
			error("Invalid input detected\n");

		}
		//if the user passed only one input image file then 
		//split the image 
		if (argv == 2) {
			//declare variable for the image buffer and read image from file
			cv::Mat data = cv::imread(argc[1]);

			//convert image to gray
			cv::cvtColor(data, data, CV_BGR2GRAY);

			//save to the container
			myParameters()->Input.push_back(data);

		}
		// check if the image file is four
		//Note: the image file has to pass in order of 00, 10, 01, 11 
		// where 10 means the angle at second row and first column looking at 2-by-2 pixel
		else if (argv == 5)

		{
			//declare variables for the image buffer and read images from file
			cv::Mat im0, im1, im2, im3;
			im0 = cv::imread(argc[1]);
			im1 = cv::imread(argc[2]);
			im2 = cv::imread(argc[3]);
			im3 = cv::imread(argc[4]);

			//convert image to gray
			cv::cvtColor(im0, im0, CV_BGR2GRAY);
			cv::cvtColor(im1, im1, CV_BGR2GRAY);
			cv::cvtColor(im2, im2, CV_BGR2GRAY);
			cv::cvtColor(im3, im3, CV_BGR2GRAY);

			//save to the container
			myParameters()->Input.push_back(im0);
			myParameters()->Input.push_back(im1);
			myParameters()->Input.push_back(im2);
			myParameters()->Input.push_back(im3);
		}


	}

	void  polarization_Parameters::calculate_S0_S1_S2(const vector<cv::Mat>& input, cv::Mat& S0, cv::Mat& S1, cv::Mat& S2)
	{
		if (!((input.size() == 1) || (input.size() == 4))) {
			error_input("calculate_S0_S1_S2()");
		}

		vector<cv::Mat> tmp; //temporal value for holding the image
		cv::Mat img0;;
		cv::Mat img45;
		cv::Mat img90;
		cv::Mat img135;

		if (input.size() == 1) {

			// the image is splitted in order of 00, 10, 01, 11 
			// where 10 means the angle at second row and first column looking at 2-by-2 pixel
			imgSplit(input.at(0), tmp);
			//set the corresponding polarization orientation
			map<int, cv::Mat>mymap;
			mymap[myParameters()->orientPolAngleR0C0] = tmp[0].clone();
			mymap[myParameters()->orientPolAngleR1C0] = tmp[1].clone();
			mymap[myParameters()->orientPolAngleR0C1] = tmp[2].clone();
			mymap[myParameters()->orientPolAngleR1C1] = tmp[3].clone();

			img0 = mymap.at(0).clone();
			img45 = mymap.at(45).clone();
			img90 = mymap.at(90).clone();
			img135 = mymap.at(135).clone();

		}

		else {
			img0 = input.at(0).clone();
			img45 = input.at(1).clone();
			img90 = input.at(2).clone();
			img135 = input.at(3).clone();
		}





		img0.convertTo(img0, CV_64FC1);
		img45.convertTo(img45, CV_64FC1);
		img90.convertTo(img90, CV_64FC1);
		img135.convertTo(img135, CV_64FC1);

		S1.create(img0.rows, img0.cols, CV_64FC1);

		S2 = S1.clone();
		S0 = S1.clone();
		cv::Mat tmp1 = S1.clone();
		cv::Mat tmp2 = S1.clone();

		//compute Stoke's vector S0
		S0 = 0.5*(img0 + img45 + img90 + img135);
		S0.convertTo(myParameters()->Intensity, CV_8UC1);

		tmp1 = (img0 - img90);
		tmp2 = (img45 - img135);

		//compute Stoke's vector S1 and S2
		cv::divide(tmp1, S0, S1);
		cv::divide(tmp2, S0, S2);
	}


	void polarization_Parameters::executePolarizationParameters() {
		if (!((myParameters()->Input.size() == 1) || (myParameters()->Input.size() == 4))) {
			error_input("executePolarizationParameters()");
		}

		//compute Stoke's vector
		cv::Mat S1;
		cv::Mat S2;
		cv::Mat S0;

		calculate_S0_S1_S2(myParameters()->Input, S0, S1, S2);

		//convert S0 to uchar
		S0.convertTo(myParameters()->Intensity, CV_8UC1);

		//calc AoP
		calculateAOP(S1, S2, myParameters()->AoLP);

		//calc DoP
		calculateDOP(S1, S2, myParameters()->DoLP);

		//calc zenith
		calcZenith(myParameters()->DoLP, myParameters()->Zenith);

		//calc azimuth
		calculateAzimuth(myParameters()->AoLP, myParameters()->Azimuth);

		//calc normal vector
		calcNormalVectorPol(myParameters()->Zenith, myParameters()->Azimuth);

	}


	void polarization_Parameters::calculateAOP(const cv::Mat& S1, const cv::Mat& S2, cv::Mat& AOP) {
		//test for input validity
		if (S2.rows != S1.rows || S2.cols != S1.cols || S2.empty() || S1.empty()) {
			error_input("calculateAOP()");
		}
		double mini, Maxim;
		AOP.release();
		AOP.create(S1.rows, S1.cols, CV_64FC1);  //convert mat to double precicion
		int rows = S1.rows;
		int cols = S1.cols;

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {

				double x = S1.at<double>(r, c);
				double y = S2.at<double>(r, c);

				double res = 0.5 * std::atan2(y, x);

				if (y < 0) {
					res += CV_PI;
				}
				AOP.at<double>(r, c) = res;

			}
		}


		cv::minMaxIdx(AOP, &mini, &Maxim);

		////scale for display purpose
		cv::Mat output = AOP.clone() * 60.0;

		output.convertTo(output, CV_8UC1);
		//save the image
		imgSave(output, "AoLP", ".png");
	}
	void polarization_Parameters::calculateDOP(const cv::Mat& S1, const cv::Mat& S2, cv::Mat& DOLP) {
		//test for input validity
		if (S2.rows != S1.rows || S2.cols != S1.cols || S2.empty() || S1.empty()) {
			error_input("calculateDOP()");
		}
		DOLP.release();
		DOLP.create(S1.rows, S1.cols, CV_64FC1);  //convert mat to double precicion

		double mini, Maxim;
		cv::Mat tmpS1, tmpS2, tmpS3, tmpS4, DOLP_tmp;
		tmpS1.convertTo(tmpS1, CV_64FC1);
		tmpS2.convertTo(tmpS2, CV_64FC1);
		tmpS3.convertTo(tmpS3, CV_64FC1);



		cv::pow(S1, 2.0, tmpS1);
		cv::pow(S2, 2.0, tmpS2);
		cv::add(tmpS1, tmpS2, tmpS3);
		cv::sqrt(tmpS3, DOLP);


		cv::minMaxIdx(DOLP, &mini, &Maxim);

		////scale for display purpose
		cv::Mat output = DOLP.clone();

		output = (output / Maxim) *800.0;
		output.convertTo(output, CV_8UC1);

		imgSave(output, "DOLP", ".png");
	}


	void polarization_Parameters::calcZenith(const cv::Mat &DoP, cv::Mat &Zenith) {
		//test for input validity
		if (DoP.empty()) {
			error_input("calcZenith()");
		}
		Zenith.release();
		Zenith.create(DoP.rows, DoP.cols, DoP.type());

		int rows = DoP.rows;
		int cols = DoP.cols;


		double aa, bb, cc, dd, retval = 0.f, t1, t2;
		double n = myParameters()->refind;
		//calc zenith

		if (v_polType == polarization_Parameters::SPECULAR) { //check for specular reflection 
			for (int r = 0; r < rows; r++) {
				const double *Pzen = Zenith.ptr<double>(r);
				const double *pDoP = DoP.ptr<double>(r);
				for (int c = 0; c < cols; c++) {
					retval = 0.f;
					double d = *pDoP;
					double q = pow(n, 2);
					aa = sqrt(pow(n, 4.0) - pow(n, 2.0) *q* pow(d, 2.0));
					bb = sqrt((8 * pow(n, 4)) - (8 * pow(n, 2) * (q - 1) * pow(d, 2)) + ((q - 4) * q * pow(d, 4)) + (8 * pow(n, 2) * aa) + (8 * pow(d, 2) * aa) - (4 * q* pow(d, 2)*aa));
					cc = sqrt(8 * pow(n, 4) - 8 * pow(n, 2)*(q - 1)*pow(d, 2) + (q - 4)*q*pow(d, 4) - 8 * pow(n, 2)*aa - 8 * pow(d, 2)*aa + 4 * q*pow(d, 2)*aa);

					t1 = (pow(n, 2)*(bb / pow(d, 2) - q) - aa * (bb / pow(d, 2) + q)) / (n*q);
					t1 = 1 / t1;
					t1 = atan(t1*sqrt(2 * bb - 4 * aa - 4 * pow(n, 2) + 2 * q*pow(d, 2)));
					//if (!(0 <= t1 && t1 <= (CV_PI / 2)))t1 = 0;
					t2 = (pow(n, 2) *(cc / pow(d, 2) - q) + aa * (cc / pow(d, 2) + q)) / (n*q);
					t2 = 1 / t2;
					t2 = atan(t2 *sqrt(2 * cc + 4 * aa - 4 * pow(n, 2) + 2 * q*pow(d, 2)));
					//if (!(0 <= t2 && t2 <= (CV_PI / 2)))t2 = 0;

					//check for valid value of ref index
					//check if both are valid but t1 < t2
					if (!isnan(t1) && !isinf(t1) && !isnan(t2) && !isinf(t2) && t1 < t2)retval = t1;
					//check if both are valid but t1 > t2
					else if (!isnan(t1) && !isinf(t1) && !isnan(t2) && !isinf(t2) && t1 > t2)retval = t2;
					//check if only t1 is valid
					else if (!isnan(t1) && !isinf(t1) && (isnan(t2) || isinf(t2)))retval = t1;
					//check if only t2 is valid
					else if (!isnan(t2) && !isinf(t2) && (isnan(t1) || isinf(t1)))retval = t2;

					Zenith.at<double>(r, c) = retval;

					++pDoP;
					++Pzen;
				}


			}

		}
		else if (v_polType == polarization_Parameters::DIFFUSE) {
			for (int r = 0; r < rows; r++) { //check for diffuse reflection
				const double *Pzen = Zenith.ptr<double>(r);
				const double *pDoP = DoP.ptr<double>(r);
				for (int c = 0; c < cols; c++) {
					retval = 0.f;
					aa = pow((n - (1 / n)), 2.0) + ((*pDoP) * pow((n + (1 / n)), 2.0));
					bb = 4 * (*pDoP) * ((n * n) + 1) * (aa - 4 * (*pDoP));
					cc = (bb * bb) + (16 * (*pDoP) * (*pDoP)) * (16 * ((*pDoP) * (*pDoP)) - (aa * aa))  * ((n * n) - 1) * ((n * n) - 1);
					dd = sqrt(((-bb - sqrt(cc)) / (2 * (16 * (*pDoP * (*pDoP)) - aa * aa))));
					retval = std::real(asin(dd));

					if (!isnan(retval) && !isinf(retval))Zenith.at<double>(r, c) = retval;
					++pDoP;
					++Pzen;
				}
			}

		}
		else error("invalid Polarization type: set either diffuse or specular");
		Zenith.convertTo(Zenith, DoP.type());
	}


	void polarization_Parameters::calculateAzimuth(const cv::Mat &AoP, cv::Mat &Azimuth) {
		//Azimuth = AoP.clone();
		//if (m_typ == "specular")

		//test for input validity
		if (AoP.empty()) {
			error_input("calculateAzimuth()");
		}
		Azimuth = AoP.clone() + (CV_PI / 2);
	}


	void polarization_Parameters::calcNormalVectorPol(const cv::Mat &Zenith, const cv::Mat& Azimuth) {
		//test for input validity
		if (Zenith.empty() || Azimuth.empty()) {
			error_input("calcNormalVectorPol()");
		}
		//declare component of normal along x,y,z direction
		cv::Mat polNx, polNy, polNz;
		int rows = Zenith.rows;
		int cols = Zenith.cols;

		polNx.create(rows, cols, Zenith.type());
		polNy = polNx.clone();
		polNz = polNx.clone();

		for (int r = 0; r < rows; r++) {
			const double *PZenith = Zenith.ptr<double>(r);
			const double *PAzimuth = Azimuth.ptr<double>(r);

			double *Pgx = polNx.ptr<double>(r);
			double *Pgy = polNy.ptr<double>(r);
			double *Pgz = polNz.ptr<double>(r);

			for (int c = 0; c < cols; c++) {
				double x, y, z;
				x = std::cos(*PAzimuth) * std::sin(*PZenith);
				y = std::sin(*PAzimuth) * std::sin(*PZenith);
				z = std::cos(*PZenith);

				*Pgx = x;
				*Pgy = y;
				*Pgz = z;

				++PZenith;
				++PAzimuth;
				++Pgx;
				++Pgy;
				++Pgz;

			}
		}
		myParameters()->Normal_x = polNx.clone();
		myParameters()->Normal_y = polNy.clone();
		myParameters()->Normal_z = polNz.clone();

	}




}

