#include "CameraUsage.h"
#include <fstream>
#include <opencv2/bgsegm.hpp>

using namespace cv;
using namespace std;

void CameraUsage::preprocess(const Mat& frame, dnn::Net& net, Size inpSize, float scale, const Scalar& mean, bool swapRB)
{
	Mat blob;
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	dnn::blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

	// Run a model.
	net.setInput(blob, "", scale, mean);
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		resize(frame, frame, inpSize);
		Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		net.setInput(imInfo, "im_info");
	}
}

void CameraUsage::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<std::string>& classes)
{
	if (conf > 1.0) {
		cout << "Too big!\n";
		return;
	}
	double factor = std::max(frame.size().height / 600.0, 1.0);
	static const size_t n = 6;
	static const array<Scalar, n> color{ Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 255, 0), Scalar(255, 255, 0), Scalar(0, 255, 255) };
	rectangle(frame, Point(left, top), Point(right, bottom), color[classId % n], (int) factor);

	std::string label = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5 * factor, (int) factor, &baseLine);

	top = max(top, labelSize.height);

	rectangle(frame, Point(left, top - labelSize.height),
		Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5 * factor, Scalar(), (int) factor);
}

void CameraUsage::postprocess(Mat& frame, const vector<Mat>& outs, dnn::Net& net, float& confThreshold, float& nmsThreshold, vector<string>& classes)
{
	std::vector<int> outLayers = net.getUnconnectedOutLayers();
	std::string outLayerType = net.getLayer(outLayers[0])->type;

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;
	if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() > 0);
		for (size_t k = 0; k < outs.size(); k++)
		{
			float* data = (float*)outs[k].data;
			for (size_t i = 0; i < outs[k].total(); i += 7)
			{
				float confidence = data[i + 2];
				if (confidence > confThreshold)
				{
					int left = (int)data[i + 3];
					int top = (int)data[i + 4];
					int right = (int)data[i + 5];
					int bottom = (int)data[i + 6];
					int width = right - left + 1;
					int height = bottom - top + 1;
					if (width <= 2 || height <= 2)
					{
						left = (int)(data[i + 3] * frame.cols);
						top = (int)(data[i + 4] * frame.rows);
						right = (int)(data[i + 5] * frame.cols);
						bottom = (int)(data[i + 6] * frame.rows);
						width = right - left + 1;
						height = bottom - top + 1;
					}
					classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
					boxes.push_back(Rect(left, top, width, height));
					confidences.push_back(confidence);
				}
			}
		}
	}
	else if (outLayerType == "Region")
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
	}
	else
		CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

	std::vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classes);
	}
}

void CameraUsage::detectObject()
{
	string modelPath = "models/yolov4.weights";
	string configPath = "models/yolov4.cfg";
	vector<string> classes;

	string file = "models/object_detection_classes_yolov3.txt";
	ifstream ifs(file.c_str());
	if (!ifs.is_open())
		CV_Error(Error::StsError, "File " + file + " not found");
	string line;
	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}
	float nmsThreshold = 0.4f;
	float scale = 0.00392f;
	Scalar mean = Scalar(0, 0, 0);
	bool swapRB = true;
	int inpWidth = 512;
	int inpHeight = 512;
	//size_t asyncNumReq = 10;
	size_t asyncNumReq = 0;
	float confThreshold = 0.5f;
	int initialConf = (int)(confThreshold * 100);

	// Load a model.
	dnn::Net net = dnn::readNetFromDarknet(configPath, modelPath);
	net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
	net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);

	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	using VoidIntAction = std::function<void(int&)>;
	VoidIntAction postProcess = [&](int& pos) {
		confThreshold = pos * 0.01f;

	};
	auto callback = [](int pos, void* data)
	{
		(*(VoidIntAction*)data)(pos);
	};
	createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback, (void*) &postProcess);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap(0);

	Mat frame;
	cap >> frame;
	string fgWindowName{ "Vordergrund" };
	namedWindow(fgWindowName, WINDOW_NORMAL);
	moveWindow(fgWindowName, 0, 400);

	string fg2WindowName{ "Vordergrund 2" };
	namedWindow(fg2WindowName, WINDOW_NORMAL);
	moveWindow(fg2WindowName, 0, 800);

	Ptr<BackgroundSubtractorMOG2> fgbg = createBackgroundSubtractorMOG2();
	Mat fgmask;

	auto kernel = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Ptr<bgsegm::BackgroundSubtractorGMG> fg2bg = bgsegm::createBackgroundSubtractorGMG();
	Mat fg2mask;

	while (!frame.empty()) {
		fgbg->apply(frame, fgmask);
		imshow(fgWindowName, fgmask);

		fg2bg->apply(frame, fg2mask);
		morphologyEx(fg2mask, fg2mask, MORPH_OPEN, kernel);
		imshow(fg2WindowName, fg2mask);
		
		preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
		std::vector<Mat> outs;
		net.forward(outs, outNames);
		if (waitKey(1) == 27)
			break;
		postprocess(frame, outs, net, confThreshold, nmsThreshold, classes);
		imshow(kWinName, frame);
		cap >> frame;
	}
	destroyAllWindows();
	return;
}
