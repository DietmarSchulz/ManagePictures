#pragma once
#include <string>
#include <array>
#include <opencv2/opencv.hpp>

class CameraUsage
{
	static const size_t n = 7;
	inline static const std::array<cv::Scalar, n> color{ cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), };

	void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
		const cv::Scalar& mean, bool swapRB);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string>& classes);
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, float& confThreshold, float& nmsThreshold, std::vector<std::string>& classes);
public:
	void detectObject();
};

