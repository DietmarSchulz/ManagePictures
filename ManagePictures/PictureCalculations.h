#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class PictureCalculations
{
	enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

	static void calcChessboardCorners(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners, Pattern patternType = CHESSBOARD);
	static cv::Mat computeHomography(const cv::Mat& R_1to2, const cv::Mat& tvec_1to2, const double d_inv, const cv::Mat& normal);
	static void computeC2MC1(const cv::Mat& R1, const cv::Mat& tvec1, const cv::Mat& R2, const cv::Mat& tvec2,
		cv::Mat& R_1to2, cv::Mat& tvec_1to2);
	static void decomposeHomography(const std::string& img1Path, const std::string& img2Path, const cv::Size& patternSize,
		const float squareSize, const std::string& intrinsicsPath);
	static cv::Scalar randomColor(cv::RNG& rng);
public:
	void AddPicture(std::string& firstPic);
	void RGBManipulation(std::string& picName);
	void RandomForests(std::string& picName);
	void Homography(std::string& picName);
	void Matches(std::string& picName);
};

