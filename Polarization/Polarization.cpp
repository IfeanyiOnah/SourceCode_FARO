#include "Polarization.h"
#include "opencv2/highgui/highgui.hpp"
#include "Demosaic.h"
namespace POL {

	polContainer *Polarization::polCon = NULL;
	pcdContainer *Polarization::pcdCon = NULL;

	Polarization::Polarization()
	{
	}


	Polarization::~Polarization()
	{
		delete polCon;
		delete pcdCon;
	}



	void Polarization::padImage(const cv::Mat input, cv::Mat &output, int pad_x, int pad_y, double val) {
	
		int rows = input.rows;
		int cols = input.cols;

		int newRows = rows + 2 * pad_y;
		int newCols = cols + 2 * pad_x;

		cv::Mat output_tmp(newRows, newCols, input.type(), cv::Scalar::all(val));
		cv::Rect ROI(pad_x, pad_y, cols, rows);
		input.copyTo(output_tmp(ROI));

		output.release();
		output = output_tmp.clone();
	}

	void  Polarization::imgSplit(const cv::Mat &input, vector<cv::Mat>&output, int method) {
		cv::Mat im00, im01, im10, im11;
		//cv::cvtColor(input, input, CV_BGR2GRAY);
		int rows = input.rows / 2;
		int cols = input.cols / 2;
		//release the entire image
		im00.release();
		im01.release();
		im10.release();
		im11.release();

		im00.create(rows, cols, input.type());
		im01.create(rows, cols, input.type());
		im10.create(rows, cols, input.type());
		im11.create(rows, cols, input.type());

		if (method == imgPolType::SEPARATED) {
			//create region of interest and copy to that region
			cv::Rect Rim10(cols, 0, cols, rows);
			input(Rim10).copyTo(im10);

			cv::Rect Rim00(0, 0, cols, rows);
			input(Rim00).copyTo(im00);

			cv::Rect Rim11(cols, rows, cols, rows);
			input(Rim11).copyTo(im11);

			cv::Rect Rim01(0, rows, cols, rows);
			input(Rim01).copyTo(im01);

		}
		else if (method == imgPolType::RAW) {
			for (int r = 0; r < rows; ++r) {
				uchar *Pim00 = im00.ptr<uchar>(r);
				uchar *Pim01 = im01.ptr<uchar>(r);
				uchar *Pim10 = im10.ptr<uchar>(r);
				uchar *Pim11 = im11.ptr<uchar>(r);
				for (int c = 0; c < cols; ++c) {
					//create mask
					cv::Mat mask(2, 2, input.type());
					cv::Rect rec(c * 2, r * 2, 2, 2);

					//copy to mask
					input(rec).copyTo(mask);

					//copy to image
					*Pim01 = mask.at<uchar>(0, 1);
					*Pim00 = mask.at<uchar>(0, 0);
					*Pim10 = mask.at<uchar>(1, 0);
					*Pim11 = mask.at<uchar>(1, 1);

					++Pim00;
					++Pim10;
					++Pim11;
					++Pim01;


				}
			}

		}
		else {
			error("Please insert the type of the image file to be splitted: 'sep' or 'raw'");
		}
		output.clear();

		output.push_back(im00.clone());
		output.push_back(im10.clone());
		output.push_back(im01.clone());
		output.push_back(im11.clone());
	}

	void Polarization::imgSave(const cv::Mat &img, const string &filepath, const string& format) {
		//test for input validity
		if (img.empty()) {
			error_input("imgSave()");
		}
		string filename = filepath + format;
		cv::imwrite(filename, img);
	}
}