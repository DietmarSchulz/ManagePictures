#include "Croppping.h"
#include <ranges>

using namespace cv;
using namespace std;

void Croppping::crop(std::string path)
{
	if (path.empty())
		return;
	Mat image = imread(path);
	
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat gradX, gradY;
	Sobel(gray, gradX, CV_32F, 1, 0, -1);
	Sobel(gray, gradY, CV_32F, 0, 1, -1);

	// subtract the y-gradient from the x-gradient
	Mat gradient;
	subtract(gradX, gradY, gradient);
	convertScaleAbs(gradient, gradient);

	Mat blurred;
	blur(gradient, blurred, Size(9, 9));

	Mat thresh;
	threshold(blurred, thresh, 90, 255, THRESH_BINARY);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(25, 25));

	Mat closed;
	morphologyEx(thresh, closed, MORPH_CLOSE, kernel);
	erode(closed, closed, Mat{}, Point(-1, -1), 4);
	dilate(closed, closed, Mat{}, Point(-1, -1), 4);

	Mat cnts;
	findContours(closed, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat c;
	cv::sort(cnts, cnts, true);
	c = cnts.at<char>(0);
}
