#include "PictureImprovement.h"
#include <iostream>
#include <fstream>
#include <array>
#include <execution>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qfiledialog.h>

// CUDA structures and methods
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::ximgproc;

void PictureImprovement::fillHoles(cv::Mat& mask)
{
	Mat mask_floodfill = mask.clone();
	floodFill(mask_floodfill, cv::Point(0, 0), Scalar(255));
	Mat mask2;
	bitwise_not(mask_floodfill, mask2);
	mask = (mask2 | mask);
}

void PictureImprovement::CreatebitMaskForLongestEdges(cv::Mat& faceBitMaskLongestEdge, cv::Mat& edges, int horizontalMid, cv::Mat& longestEdge)
{
	// Initialize with ellipse:
	Point center(faceBitMaskLongestEdge.size().width / 2 + 1, faceBitMaskLongestEdge.size().height / 2);
	ellipse(faceBitMaskLongestEdge, center, Size(edges.size().width / 2.7, edges.size().height / 2.6), 0, 0, 360, Scalar(255), -1); // -1 to fill

	// from left to mid in all rows
	for (auto m = 0; m < faceBitMaskLongestEdge.size().height; m++) {
		auto edgeFound = false;
		for (auto n = 0; n <= horizontalMid; n++) {
			if (longestEdge.at<unsigned char>(m, n) > 0) {
				edgeFound = true;
			}
			if (edgeFound) {
				faceBitMaskLongestEdge.at<unsigned char>(m, n) = 255;
			}
		}
	}

	// from right to mid in all rows
	for (auto m = 0; m < faceBitMaskLongestEdge.size().height; m++) {
		auto edgeFound = false;
		for (auto n = faceBitMaskLongestEdge.size().width - 1; n >= horizontalMid; n--) {
			if (longestEdge.at<unsigned char>(m, n) > 0) {
				edgeFound = true;
			}
			if (edgeFound) {
				faceBitMaskLongestEdge.at<unsigned char>(m, n) = 255;
			}
		}
	}
	auto verticalMid = faceBitMaskLongestEdge.size().height / 2 + 1;
	// from vertical Mid to Top
	for (auto n = 0; n < faceBitMaskLongestEdge.size().width; n++) {
		auto edgeFound = false;
		for (auto m = verticalMid; m >= 0; m--) {
			if (longestEdge.at<unsigned char>(m, n) > 0) {
				edgeFound = true;
			}
			if (!edgeFound) {
				// After the edge we change nothing! But else: We propagate last line in up direction
				faceBitMaskLongestEdge.at<unsigned char>(m, n) = faceBitMaskLongestEdge.at<unsigned char>(m + 1, n);
			}
		}
	}
}

void PictureImprovement::CropFace(const size_t& i, cv::Mat& faceROI, cv::Mat& ioMat, vector<Rect>& faces)
{
	string plainGray{ "Gesicht ohne eq " + to_string(i) };
	namedWindow(plainGray, WINDOW_NORMAL);
	moveWindow(plainGray, 800, 0);

	string plainGrayEq{ "Gesicht" + to_string(i) };
	namedWindow(plainGrayEq, WINDOW_NORMAL);
	moveWindow(plainGrayEq, 800, 400);

	imshow(plainGray, faceROI);
	equalizeHist(faceROI, faceROI); // Just a try to get it lighter

	imshow(plainGrayEq, faceROI);

	string randomWindowName{ "Random Forest " + to_string(i) };
	namedWindow(randomWindowName, WINDOW_NORMAL);
	moveWindow(randomWindowName, 800, 800);

	// Try to get a face cropped as bitmask
	Mat colorFace = ioMat(faces[i]).clone();
	Mat orgFloat;
	colorFace.convertTo(orgFloat, DataType<float>::type, 1 / 255.0);

	Mat edges(orgFloat.size(), orgFloat.type());

	string edgesWindowName{ "Kanten " + to_string(i) };
	namedWindow(edgesWindowName, WINDOW_NORMAL);
	moveWindow(edgesWindowName, 1200, 800);

	pDollar->detectEdges(orgFloat, edges);
	imshow(edgesWindowName, edges);

	// computes orientation from edge map
	Mat orientation_map;
	pDollar->computeOrientation(edges, orientation_map);

	// suppress edges
	Mat edge_nms;
	pDollar->edgesNms(edges, orientation_map, edge_nms, 2, 0, 1, true);

	imshow(randomWindowName, edge_nms);

	string threshedEdgeWindowName{ "Kanten gekappt " + to_string(i) };
	namedWindow(threshedEdgeWindowName, WINDOW_NORMAL);
	moveWindow(threshedEdgeWindowName, 1400, 800);

	Mat threshedEdges;
	Mat edge_nmsByte;
	edge_nms.convertTo(edge_nmsByte, DataType<unsigned char>::type, 255.0);
	threshold(edge_nmsByte, threshedEdges, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow(threshedEdgeWindowName, threshedEdges);

	for (auto m = 0; m < faceROI.size().height; m++) {
		for (auto n = 0; n < faceROI.size().width; n++) {
			if (edge_nms.at<float>(m, n) > 0.2f)
				colorFace.at<cv::Vec3b>(m, n) = { 255, 255, 255 };
		}
	}
	string colorBitmaskWindowName{ "Schwellwert Farb" + to_string(i) };
	namedWindow(colorBitmaskWindowName, WINDOW_NORMAL);
	moveWindow(colorBitmaskWindowName, 800, 400);
	imshow(colorBitmaskWindowName, colorFace);

	// Bitmaks from edges_nms:
	Mat faceBitMask = Mat::zeros(size(edge_nms), CV_8UC1);
	auto horizontalMid = faceBitMask.size().width / 2 + 1;

	// from left to mid in all rows
	for (auto m = 0; m < faceBitMask.size().height; m++) {
		auto edgeFound = false;
		for (auto n = 0; n <= horizontalMid; n++) {
			if (edge_nms.at<float>(m, n) > edgeThreshold) {
				edgeFound = true;
			}
			if (edgeFound) {
				faceBitMask.at<unsigned char>(m, n) = 255;
			}
		}
	}

	// from right to mid in all rows
	for (auto m = 0; m < faceBitMask.size().height; m++) {
		auto edgeFound = false;
		for (auto n = faceBitMask.size().width - 1; n >= horizontalMid; n--) {
			if (edge_nms.at<float>(m, n) > edgeThreshold) {
				edgeFound = true;
			}
			if (edgeFound) {
				faceBitMask.at<unsigned char>(m, n) = 255;
			}
		}
	}

	string bitmaskWindowName{ "Gesichtsbitmask " + to_string(i) };
	namedWindow(bitmaskWindowName, WINDOW_NORMAL);
	moveWindow(bitmaskWindowName, 800, 600);
	imshow(bitmaskWindowName, faceBitMask);

	auto contEdges = continuousEdges(edge_nms);
	//auto contEdges = continuousEdges(edges);
	std::ranges::sort(contEdges, [](auto& fst, auto& snd) {
		return fst.size() > snd.size();
	});
	if (contEdges.size() > 0) {
		string edgeWindowName{ "Laengste Randlinie " + to_string(i) };
		namedWindow(edgeWindowName, WINDOW_NORMAL);
		moveWindow(edgeWindowName, 800, 600);

		Mat longestEdge = Mat::zeros(size(edge_nms), CV_8UC1);
		auto numPoints{ 0 };
		auto k{ 0 };
		while (k < contEdges.size() && numPoints < 2 * edge_nms.size().height +  0.8f * edge_nms.size().width) {
			for (auto p : contEdges[k]) {
				longestEdge.at<unsigned char>(p.x, p.y) = 255;
			}
			numPoints += (int) contEdges[k].size();
			k++;
		}
		cout << k << " Kanten werden benutzt.\n";
		imshow(edgeWindowName, longestEdge);


		string bitmaskLongestEdgeWindowName{ "Gesichtsbitmask nach laengster Kannte " + to_string(i) };
		namedWindow(bitmaskLongestEdgeWindowName, WINDOW_NORMAL);
		moveWindow(bitmaskLongestEdgeWindowName, 1200, 0);

		// Bitmaks from longest edge:
		Mat faceBitMaskLongestEdge = Mat::zeros(size(longestEdge), CV_8UC1);

		CreatebitMaskForLongestEdges(faceBitMaskLongestEdge, edges, horizontalMid, longestEdge);

		imshow(bitmaskLongestEdgeWindowName, faceBitMaskLongestEdge);

		Mat bgrBitMask;
		cvtColor(faceBitMaskLongestEdge, bgrBitMask, COLOR_GRAY2BGR);

		string faceWindow { "Gesicht nach laengster Kannte ausgeschnitten " + to_string(i) }; 
		Mat cutFace;
		bitwise_and(ioMat(faces[i]), bgrBitMask, colorFace);

		namedWindow(faceWindow, WINDOW_NORMAL);
		imshow(faceWindow, colorFace);
	}
}

void PictureImprovement::findNearest(float pixelValue, cv::Point& nearestPoint, PointSet_t& contEdge, int& m, int& n)
{
	if (contEdge.size() == 0)
		return;
	PointSet_t::const_iterator nearestPointIt;
	int mask[8][2]{
		{-1, -1}, {-1, 0}, {-1, 1},
		{ 0, -1},          { 0, 1},
		{ 1, -1}, { 1, 0}, { 1, 1} };
	for (auto i = 0; i <= neighborhoodThreshold; i++) {
		for (auto j = 0; j <= neighborhoodThreshold; j++) {
			if (pixelValue - (i + j - 2) * edgeThreshold < 0.0)
				break;
			for (auto mul : mask) {
				nearestPointIt = contEdge.find(Point(m + i * mul[0], n + j * mul[1]));
				if (nearestPointIt != contEdge.end()) {
					nearestPoint = *nearestPointIt;
					return;
				}
			}
		}
	}
}

int PictureImprovement::MergePointsInSets(cv::Mat& edges, int& m, int& n, std::vector<PointSet_t>& res)
{
	auto numMerges{ 0 };
	if (edges.at<float>(m, n) > edgeThreshold) {
		auto mergeThem{ -1 };
		for (auto i = 0; auto & contEdge : res) {
			auto pointIt = contEdge.find(Point{ m, n });
			if (pointIt != contEdge.end()) {
				if (mergeThem != -1) {
					contEdge.insert(res[mergeThem].begin(), res[mergeThem].end());
					res[mergeThem].clear();
					numMerges++;
				}
				mergeThem = i;
			}
			i++;
		}
	}
	return numMerges;
}

void PictureImprovement::InsertPointInSets(cv::Mat& edges, int& m, int& n, std::vector<PointSet_t>& res)
{
	auto currentPixel = edges.at<float>(m, n);
	auto added{ false };
	for_each(execution::par, res.begin(), res.end(), [&added, &m, &n, &currentPixel, this](auto& contEdge) {
		Point nearestPoint{ -neighborhoodThreshold - 1, -neighborhoodThreshold - 1 }; // intialized outside neighbourhood
		findNearest(currentPixel, nearestPoint, contEdge, m, n);

		// Is that last point in neigborhood of current edge:
		if (abs(m - nearestPoint.x) <= neighborhoodThreshold && abs(n - nearestPoint.y) <= neighborhoodThreshold) {
			if (auto xdiff = abs(m - nearestPoint.x), ydiff = abs(n - nearestPoint.y); xdiff > 1 || ydiff > 1) {
				// Points to fill the gap are needed:
				while (xdiff > 1 || ydiff > 1) {
					if (xdiff == 1) {
						if (nearestPoint.y > n) {
							nearestPoint.y--;
						}
						else {
							nearestPoint.y++;
						}
						ydiff--;
					}
					else if (ydiff == 1) {
						if (nearestPoint.x > m) {
							nearestPoint.x--;
						}
						else {
							nearestPoint.x++;
						}
						xdiff--;
					}
					else {
						if (nearestPoint.y > n) {
							nearestPoint.y--;
						}
						else {
							nearestPoint.y++;
						}
						if (nearestPoint.x > m) {
							nearestPoint.x--;
						}
						else {
							nearestPoint.x++;
						}
						xdiff--; ydiff--;
					}
					contEdge.insert(nearestPoint);
				}
			}
			contEdge.insert(Point(m, n));
			added = true;
		}
	});
	if (!added) {
		res.emplace_back(PointSet_t{ Point2i(m,n) });
	}
}

std::vector<PictureImprovement::PointSet_t> PictureImprovement::continuousEdges(cv::Mat& edges)
{
	vector<PointSet_t> res;
	vector<pair<int, int>> lightPoints;

	// Let's build up a filled ellipse with a certain distance to edges' borders 
	// There we don't expect the face's edges
	Mat filledEllipseMask = Mat::zeros(size(edges), CV_8UC1);

	Point center(edges.size().width / 2 + 1, edges.size().height / 2);
	ellipse(filledEllipseMask, center, Size(edges.size().width / 2.7, edges.size().height / 2.6), 0, 0, 360, Scalar(255), -1); // -1 to fill

	// look forward in both directions
	for (auto m = 0; m < edges.size().height; m++) {
		for (auto n = 0; n < edges.size().width; n++) {
			if (filledEllipseMask.at<unsigned char>(m, n) > 0)
				continue;
			if (edges.at<float>(m, n) > 0.0f) {
				lightPoints.emplace_back(m, n);
				InsertPointInSets(edges, m, n, res);
			}
		}
	}
	cout << "1. Einsammeln fertig mit " << res.size() << " sets\n";

	auto numMerges{ 0 };
	// look forward in rows and backward in columns
	for (auto& p : lightPoints){
		numMerges += MergePointsInSets(edges, p.first, p.second, res);
	}
	cout << "1. Merges: " << numMerges << "\n";
	std::erase_if(res, [](auto& contEdge) { return contEdge.size() == 0; });

	return res;
}

void PictureImprovement::redEyeRemoving(Mat& ioMat)
{
	Mat frame_gray;
	cvtColor(ioMat, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		if (showCircles) {
			ellipse(ioMat, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
			rectangle(ioMat, faces[i], Scalar(255, 255, 255), 4);
		}
		Mat faceROI = frame_gray(faces[i]);
		CropFace(i, faceROI, ioMat, faces);
		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, CASCADE_SCALE_IMAGE);
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			if (showCircles)
				circle(ioMat, eye_center, radius, Scalar(255, 0, 0), 4);

			// Extract eye from the image.
			auto currEye = eyes[j];
			currEye.x += faces[i].x;
			currEye.y += faces[i].y;
			Mat eye = ioMat(currEye);

			// Split eye image into 3 channels. 
			vector<Mat>bgr(3);
			split(eye, bgr);

			// Simple red eye detector
			Mat mask = bgr[0] < 0;
			circle(mask, Point(eyes[j].width / 2, eyes[j].height / 2), radius / 3.0, Scalar(255, 255, 255, 255), -1);
			mask = mask & (((bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]))) | ((bgr[2] > 240) & (bgr[0] < 180) & (bgr[1] < 180)));
			fillHoles(mask);
			dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

			// Calculate the mean channel by averaging
			// the green and blue channels
			Mat mean = (bgr[0] + bgr[1]) / 2;

			// Copy the mean image to blue channel with mask.
			mean.copyTo(bgr[0], mask);

			// Copy the mean image to green channel with mask.
			mean.copyTo(bgr[1], mask);

			// Copy the mean image to red channel with mask.
			mean.copyTo(bgr[2], mask);

			// Merge the three channels
			Mat eyeOut;
			merge(bgr, eyeOut);

			// Copy the fixed eye to the output image. 
			eyeOut.copyTo(ioMat(currEye));
		}
	}
}

void PictureImprovement::ExtractFaces(cv::Mat& ioMat)
{
	Mat frame_gray;
	cvtColor(ioMat, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3);
	for (auto i = 0; i < faces.size(); i++)
	{
		if (faces[i].width < 100 && faces[i].height < 100)
			continue; // Discard too small objects
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		if (showCircles) {
			ellipse(ioMat, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
			rectangle(ioMat, faces[i], Scalar(255, 255, 255), 4);
		}
		Mat faceROI = frame_gray(faces[i]);
		string grayFaceRegion = "Image containing Face " + to_string(i);
		namedWindow(grayFaceRegion, WINDOW_NORMAL);
		moveWindow(grayFaceRegion, i * 400, 0);
		imshow(grayFaceRegion, faceROI);

		// the models (internally used)
		Mat bgModel, fgModel;
		// segmentation result
		Mat result; // segmentation (4 possible values)

		// GrabCut segmentation
		int border = faces[i].width / 4;
		Rect rectangle(faces[i].x - border, faces[i].y - border, faces[i].width + 2 * border, faces[i].height + 2 * border);
		grabCut(ioMat,    // input image
			result,   // segmentation result
			rectangle,// rectangle containing foreground 
			bgModel, fgModel, // models
			5,        // number of iterations
			cv::GC_INIT_WITH_RECT); // use rectangle

		// Get the pixels marked as likely foreground
		compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
		// or:
		//	result= result&1;

		// create a white image
		cv::Mat foreground(ioMat.size(), CV_8UC3,
			cv::Scalar(255, 255, 255));

		ioMat.copyTo(foreground, result); // bg pixels not copied

		string faceString = "Face " + to_string(i);
		namedWindow(faceString, WINDOW_NORMAL);
		moveWindow(faceString, i * 400, 400);
		imshow(faceString, foreground);
	}
}

void PictureImprovement::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes, size_t index)
{
	if (conf > 1.0) {
		cout << "Too big!\n";
		return;
	}
	double factor = std::max(frame.size().height / 600.0, 1.0);
	static const size_t n = 6;
	static const array<Scalar, n> color{ Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 255, 0), Scalar(255, 255, 0), Scalar(0, 255, 255) };
	rectangle(frame, Point(left, top), Point(right, bottom), color[index % n], factor);

	std::string label = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5 * factor, factor, &baseLine);

	top = max(top, labelSize.height);

	rectangle(frame, Point(left, top - labelSize.height),
		Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5 * factor, Scalar(), factor);
}

Net PictureImprovement::prepareObjectRecognition(vector<string>& classes)
{
	string modelPath = "models/yolov4.weights";
	string configPath = "models/yolov4.cfg";

	string file = "models/object_detection_classes_yolov3.txt";
	ifstream ifs(file.c_str());
	if (!ifs.is_open())
		CV_Error(Error::StsError, "File " + file + " not found");
	string line;
	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}

	// Load a model.
	dnn::Net net = dnn::readNetFromDarknet(configPath, modelPath);
	net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
	net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);

	return net;
}

void PictureImprovement::preprocess(const Mat& frame, Net& net, Size inpSize, float scale, const Scalar& mean, bool swapRB)
{
	Mat blob;
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

	// Run a model.
	net.setInput(blob, "", scale, mean);
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		resize(frame, frame, inpSize);
		Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		net.setInput(imInfo, "im_info");
	}
}

void PictureImprovement::postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, float& confThreshold, float& nmsThreshold, vector<string>& classes)
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
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classes, i);
	}
}

void PictureImprovement::erode(cv::Mat& ioMat)
{
	if (doErosion) {
		// 0 MORPH_RECT
		// 1 MORPH_CROSS
		// 2 MORPH_ELLIPSE
		Mat element = getStructuringElement(morphType,
			Size(2 * kernelSize + 1, 2 * kernelSize + 1),
			Point(kernelSize, kernelSize));
		cv::erode(ioMat, ioMat, element);
	}
}

PictureImprovement::PictureImprovement()
{
	String face_cascade_name = "models/haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "models/haarcascade_eye_tree_eyeglasses.xml";

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return;
	};
	pDollar = createStructuredEdgeDetection("models/model.yml");
}

void PictureImprovement::elementary(std::string& s)
{
	Mat imageSource;
	string windowNameOrg = "Original image";
	string windowNameGamma = "Gamma window";

	imageSource = imread(s);

	if (imageSource.empty())
		return;

	// Filter:
	bool doFilter{ false };
	bool doCascade{ false };
	bool doGaussian{ false };
	int kernelSize{ 3 };
	int kernelStep{ 1 };
	bool doRecognition{ false };
	bool doExtractFaces{ false };

	// Gamma brightness:
	int gammaI = 100;
	double gamma_ = gammaI / 100.0;
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
	Mat imageDestination;

	using VoidAction = std::function<void()>;

	TrackbarCallback callbackForTrackBar = [](int pos, void* userdata)
	{
		(*(VoidAction*)userdata)();
	};

	VoidAction doLaplace = [&] {
		if (doFilter) { 
			Mat kernel = (Mat_<int>(3, 3) <<
				1, 1, 1,
				1, -8, 1,
				1, 1, 1); // an approximation of second derivative, a quite strong kernel
			//cuda::GpuMat gKernel;
			//gKernel.upload(kernel);

			Mat imgLaplacian;
			//cuda::GpuMat gImgLaplacian;

			filter2D(imageDestination, imgLaplacian, CV_32F, kernel);
			//Ptr<cuda::Filter> lapl = cuda::createLaplacianFilter(CV_32F, CV_32F);
			
			Mat sharp;
			//cuda::GpuMat gSharp;
			imageDestination.convertTo(sharp, CV_32F);
			Mat imgResult = sharp - imgLaplacian;
			namedWindow("Laplace", WINDOW_NORMAL);
			imgLaplacian.convertTo(imgLaplacian, CV_8U);
			imshow("Laplace", imgLaplacian);
			// convert back to 8bits gray scale
			imgResult.convertTo(imageDestination, CV_8UC3);
			imshow(windowNameGamma, imageDestination);
		} 
	};

	vector<string> classes;
	Net net = prepareObjectRecognition(classes);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
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

	VoidAction doGammaLUT = [&]() {
		gamma_ = gammaI / 100.0;
		for (int i = 0; i < 256; ++i)
			p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
		LUT(imageSource, lookUpTable, imageDestination);
		doLaplace();
		erode(imageDestination);
		if (doGaussian) {
			kernelSize = kernelStep * 2 + 1;
			GaussianBlur(imageDestination, imageDestination, Size(kernelSize, kernelSize), 0, 0);
		}
		if (doCascade)
			redEyeRemoving(imageDestination);
		if (doExtractFaces)
			ExtractFaces(imageDestination);
		if (doRecognition) {
			preprocess(imageDestination, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
			std::vector<Mat> outs;
			net.forward(outs, outNames);

			postprocess(imageDestination, outs, net, confThreshold, nmsThreshold, classes);
		}
		if (!doFilter || doCascade || doGaussian || doRecognition || doErosion)
			imshow(windowNameGamma, imageDestination);
	};

	ButtonCallback callbackForSave = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};

	VoidAction checkSharpen = [&]() {
		doFilter = !doFilter;
		doLaplace();
		if (!doFilter)
			doGammaLUT();
	};

	VoidAction doSaveAs = [&]() {
		cout << "Save:\n";
		string saveAs = QFileDialog::getSaveFileName(nullptr, "Save as ..", QString(), "All picture Files (*.jpg *.png *.tiff)").toStdString();
		if (saveAs.empty()) {
			cout << "nothing\n";
			return;
		}
		else {
			cout << saveAs << "\n";
			imwrite(saveAs, imageDestination);
		}

	};

	namedWindow(windowNameOrg, WINDOW_NORMAL);
	namedWindow(windowNameGamma, WINDOW_NORMAL);
	moveWindow(windowNameOrg, 0, 0);
	moveWindow(windowNameGamma, 400, 0);
	imshow(windowNameOrg, imageSource);

	createTrackbar("Gamma", windowNameGamma, &gammaI, 200, callbackForTrackBar, (void*)&doGammaLUT);
	createButton("Save", callbackForSave, (void*)&doSaveAs);

	int iDummy = 0;

	ButtonCallback callbackForFilter = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};

	createButton("Sharpen", callbackForFilter, (void*)&checkSharpen, QT_CHECKBOX | QT_NEW_BUTTONBAR, false);

	ButtonCallback callbackForRedEyes = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};
	VoidAction checkCascade = [&]() {
		doCascade = !doCascade;
		doGammaLUT();
	};
	createButton("Remove red eyes", callbackForRedEyes, (void*)&checkCascade, QT_CHECKBOX | QT_NEW_BUTTONBAR, false);

	ButtonCallback callbackForExtractFaces = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};
	VoidAction checkExtractFaces = [&]() {
		doExtractFaces = !doExtractFaces;
		doGammaLUT();
	};
	createButton("Extract faces", callbackForExtractFaces, (void*)&checkExtractFaces, QT_CHECKBOX | QT_NEW_BUTTONBAR, false);

	ButtonCallback callbackForCircles = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};
	VoidAction doCircles = [&]() {
		showCircles = !showCircles;
		doGammaLUT();
	};
	createButton("Show cirles", callbackForCircles, (void*) &doCircles, QT_CHECKBOX, false);

	ButtonCallback callbackForGaussian = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};
	VoidAction actionGaussian = [&]() {
		doGaussian = !doGaussian;
		doGammaLUT();
	};
	createButton("Gaussian blur", callbackForGaussian, (void*)&actionGaussian, QT_CHECKBOX, false);

	TrackbarCallback callbackForTrackBarKernel = [](int pos, void* userdata)
	{
		(*(VoidAction*)userdata)();
	};

	VoidAction doKernel = [&] {
		doGammaLUT();
	};
	createTrackbar("Kernel size", "", &kernelStep, 20, callbackForTrackBarKernel, (void*)&doKernel);

	ButtonCallback callbackForRecognition = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};
	VoidAction actionRecognition = [&]() {
		doRecognition = !doRecognition;
		doGammaLUT();
	};
	createButton("Object recognition", callbackForRecognition, (void*)&actionRecognition, QT_CHECKBOX, false);

	using VoidIntAction = std::function<void(int&)>;

	VoidIntAction postProcess = [&](int& pos) {
		confThreshold = pos * 0.01f;

	};

	auto callback = [](int pos, void* data)
	{
		(*(VoidIntAction*)data)(pos);
	};

	createTrackbar("Confidence threshold, %", "", &initialConf, 99, callback, (void*)&postProcess);

	ButtonCallback callbackForErosion = [](int state, void* userdata) {
		(*(VoidAction*)userdata)();
	};
	VoidAction actionErosion = [&]() {
		doErosion = !doErosion;
		doGammaLUT();
	};
	createButton("Image erosion", callbackForErosion, (void*)&actionErosion, QT_CHECKBOX, false);

	VoidIntAction Erosion = [&](int& pos) {
		doGammaLUT();
	};

	createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "",
		&this->morphType, max_morph, callback,
		(void*) &Erosion);
	createTrackbar("Kernel size:\n 2n +1", "",
		&this->kernelSize, max_kernel_size, callback,
		(void*) &Erosion);

	callbackForTrackBar(iDummy, (void*)&doGammaLUT);

	auto wait_time = 1000;
	while (getWindowProperty(windowNameGamma, WND_PROP_VISIBLE) >= 1) {
		auto keyCode = waitKey(wait_time);
		if (keyCode == 27) { // Wait for ESC key stroke
			destroyAllWindows();
			break;
		}
	}
	destroyAllWindows();
	cout << "Output finished!\n";
}
