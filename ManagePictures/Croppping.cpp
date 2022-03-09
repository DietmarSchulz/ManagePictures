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
	erode(closed, closed, Mat{}, Point(-1, -1), 10);
	dilate(closed, closed, Mat{}, Point(-1, -1), 20);

	vector<vector<Point>> cnts;
	findContours(closed, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<Point> c;
	ranges::sort(cnts, [](const auto& v1, const auto& v2) { return contourArea(v1) > contourArea(v2); });
	c = cnts[0];

	// compute the rotated bounding box of the largest contour
	auto rect = minAreaRect(c);
	vector<vector<Point> >hull(1);
	convexHull(c, hull[0]);
	drawContours(image, hull, -1, Scalar(0, 255, 0), 3);

	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", image);

	Mat cropped;
	cv::bitwise_and(image, image, cropped, closed);
	namedWindow("Cropped Image", WINDOW_NORMAL);
	imshow("Cropped Image", cropped);
	waitKey(0);
	destroyAllWindows();
}
