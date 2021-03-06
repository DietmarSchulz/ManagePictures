#pragma once

#include <filesystem>
#include <string>
#include <map>
#include <unordered_set>

#include <opencv2/opencv.hpp>

class PictureImprovement
{
	bool showCircles{ false };
	bool doErosion{ false };
	int kernelSize = 0;
	int morphType = 0;
	int const max_morph = 2;
	int const max_kernel_size = 21;

	static void fillHoles(cv::Mat& mask);
	void redEyeRemoving(cv::Mat& ioMat);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string>& classes, size_t index);
	cv::dnn::Net prepareObjectRecognition(std::vector<std::string>& classes);
	inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
		const cv::Scalar& mean, bool swapRB);
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, float& confThreshold, float& nmsThreshold, std::vector<std::string>& classes);
	void erode(cv::Mat& ioMat);
public:
	void elementary(std::string& s);
};

