#pragma once

#include <filesystem>
#include <string>
#include <map>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

class PictureImprovement
{
public:
	// for set
	struct PointCompare
	{
		bool operator()(const cv::Point& lhs, const cv::Point& rhs) const
		{
			return (lhs.x != rhs.y) ? lhs.x < rhs.x : lhs.y < rhs.y;
		}
	};

	// for unordered_set
	struct PointHash {
		std::size_t operator()(const cv::Point& p) const {
			return std::hash<int>()((p.x << 16) + p.y);
		}
	};

	//using PointSet_t = std::unordered_set<cv::Point, PointCompare>;
	using PointSet_t = std::unordered_set<cv::Point, PointHash>;

private:
	bool showCircles{ false };
	bool doErosion{ false };
	int kernelSize = 0;
	int morphType = 0;
	int const max_morph = 2;
	int const max_kernel_size = 21;
	int neighborhoodThreshold{ 4 };
	float edgeThreshold{ 0.10f };
	cv::CascadeClassifier face_cascade;
	cv::CascadeClassifier eyes_cascade;
	cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar;

	static void fillHoles(cv::Mat& mask);
	void CreatebitMaskForLongestEdges(cv::Mat& faceBitMaskLongestEdge, cv::Mat& edges, int horizontalMid, cv::Mat& longestEdge);
	void CropFace(const size_t& i, cv::Mat& faceROI, cv::Mat& ioMat, std::vector<cv::Rect>& faces);
	//void findNearest(cv::Point& nearestPoint, PointSet_t& contEdge, int& m, int& n);
	void findNearest(float pixelValue, cv::Point& nearestPoint, PointSet_t& contEdge, int& m, int& n);
	int MergePointsInSets(cv::Mat& edges, int& m, int& n, std::vector<PointSet_t>& res);
	//void InsertPointToVectors(cv::Mat& edges, int& m, int& n, std::vector<PointSet_t>& res);
	void InsertPointInSets(cv::Mat& edges, int& m, int& n, std::vector<PointSet_t>& res);
	//std::vector<PointSet_t> continuousEdges(cv::Mat& edges);
	std::vector<PointSet_t> continuousEdges(cv::Mat& edges);
	void redEyeRemoving(cv::Mat& ioMat);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string>& classes, size_t index);
	cv::dnn::Net prepareObjectRecognition(std::vector<std::string>& classes);
	inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
		const cv::Scalar& mean, bool swapRB);
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, float& confThreshold, float& nmsThreshold, std::vector<std::string>& classes);
	void erode(cv::Mat& ioMat);
public:
	PictureImprovement();
	void elementary(std::string& s);
};

