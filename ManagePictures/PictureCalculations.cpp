#include "PictureCalculations.h"
#include <opencv2/bioinspired.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <QtWidgets/qfiledialog.h>
#include <QtWidgets/qmessagebox.h>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
using namespace cv::line_descriptor;

void PictureCalculations::AddPicture(std::string& firstPic)
{
	string firstOriginalWindowName{ "First original" };
	namedWindow(firstOriginalWindowName, WINDOW_NORMAL);
	moveWindow(firstOriginalWindowName, 0, 0);
	Mat firstImg = imread(firstPic);
	if (firstImg.empty())
		return;
	imshow(firstOriginalWindowName, firstImg);

	string secondPic = QFileDialog::getOpenFileName(nullptr, "Second picture for sum", QString(), "All picture Files (*.jpg *.png *.tiff)").toStdString();
	if (secondPic.empty()) {
		cout << "nothing\n";
		destroyAllWindows();
		return;
	}

	string secondOriginalWindowName{ "Second original" };
	namedWindow(secondOriginalWindowName, WINDOW_NORMAL);
	moveWindow(secondOriginalWindowName, 0, 320);
	Mat secondImg = imread(secondPic);
	imshow(secondOriginalWindowName, secondImg);

	if (firstImg.size() != secondImg.size()) {
		cout << "Image sizes not identical, resize to bigger one!\n";
		if (firstImg.size().width < secondImg.size().width && firstImg.size().height < secondImg.size().height) {
			resize(firstImg, firstImg, secondImg.size());
		}
		else if (firstImg.size().width > secondImg.size().width && firstImg.size().height > secondImg.size().height) {
			resize(secondImg, secondImg, firstImg.size());
		}
		else {
			cout << "Hä?";
			resize(firstImg, firstImg, secondImg.size());
		}
	}

	double alpha = 0.5; double beta;
	using VoidAction = std::function<void()>;
	String firstAndSecondAdded{ "First and second picture added" };

	//Create trackbar to change blend
	int iSliderValue1 = 0;
	namedWindow(firstAndSecondAdded, WINDOW_NORMAL);
	moveWindow(firstAndSecondAdded, 400, 0);

	VoidAction doBlending = [&] {
		//Change the brightness and contrast of the image (For more infomation http://opencv-srf.blogspot.com/2013/07/change-contrast-of-image-or-video.html)
		Mat dst;
		alpha = (double)iSliderValue1 / 100;

		beta = (1.0 - alpha);
		addWeighted(firstImg, alpha, secondImg, beta, 0.0, dst);

		//show the brightness and contrast adjusted image
		imshow(firstAndSecondAdded, dst);
	};

	TrackbarCallback callbackForTrackBar = [](int pos, void* userdata)
	{
		(*(VoidAction*)userdata)();
	};
	createTrackbar("Blend", firstAndSecondAdded, &iSliderValue1, 100, callbackForTrackBar, (void*) &doBlending);

	int iDummy{ 0 };
	callbackForTrackBar(iDummy, (void*)&doBlending);
	
	auto wait_time = 1000;
	while (getWindowProperty(firstAndSecondAdded, WND_PROP_VISIBLE) >= 1) {
		auto keyCode = waitKey(wait_time);
		if (keyCode == 27) { // Wait for ESC key stroke
			destroyAllWindows();
			break;
		}
	}

	destroyAllWindows(); //destroy all open windows
}

void PictureCalculations::RGBManipulation(string& picName)
{
	Mat orgImg = imread(picName);
	if (orgImg.empty()) {
		cout << "no Pic selected!\n";
		return;
	}
	string orgWindowName{ "Originabild" };
	namedWindow(orgWindowName, WINDOW_NORMAL);
	moveWindow(orgWindowName, 0, 0);
	imshow(orgWindowName, orgImg);

	string redWindowName{ "Rotanteil" };
	namedWindow(redWindowName, WINDOW_NORMAL);
	moveWindow(redWindowName, 400, 0);
	string greenWindowName{ "Gruenanteil" };
	namedWindow(greenWindowName, WINDOW_NORMAL);
	moveWindow(greenWindowName, 800, 0);
	string blueWindowName{ "Blauanteil" };
	namedWindow(blueWindowName, WINDOW_NORMAL);
	moveWindow(blueWindowName, 1200, 0);

	vector<Mat> bgr;
	split(orgImg, bgr);

	Mat blueDummy;
	Mat greenDummy;
	Mat redDummy;
	Mat z = Mat::zeros(bgr[0].size(), bgr[0].type());
	vector<Mat> dummy{ z, z, z };

	using VoidAction = std::function<void()>;

	// Gamma brightnesses for rgb:
	int redGammaI = 100;
	double redGamma_ = redGammaI / 100.0;
	Mat redlookUpTable(1, 256, CV_8U);
	uchar* redp = redlookUpTable.ptr();
	int greenGammaI = 100;
	double greenGamma_ = greenGammaI / 100.0;
	Mat greenlookUpTable(1, 256, CV_8U);
	uchar* greenp = greenlookUpTable.ptr();
	int blueGammaI = 100;
	double blueGamma_ = blueGammaI / 100.0;
	Mat bluelookUpTable(1, 256, CV_8U);
	uchar* bluep = bluelookUpTable.ptr();

	string resultWindowName{ "Mischergebnis" };
	namedWindow(resultWindowName, WINDOW_NORMAL);
	moveWindow(resultWindowName, 400, 400);

	auto res = QMessageBox::question(nullptr, "Mit Retina Parvo- und Magno-cellularem Bild?", 
		"Sollen die Analysen der Retina ermittelt werden?", QMessageBox::StandardButton::Yes, QMessageBox::StandardButton::No);

	if (res == QMessageBox::StandardButton::Yes) {
		// create a retina instance with default parameters setup, uncomment the initialisation you wanna test
		cv::Ptr<cv::bioinspired::Retina> myRetina;

		auto useLogSampling = false;

		// if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
		if (useLogSampling)
		{
			myRetina = cv::bioinspired::Retina::create(orgImg.size(), true, cv::bioinspired::RETINA_COLOR_BAYER, true, 2.0, 10.0);
		}
		else// -> else allocate "classical" retina :
		{
			myRetina = cv::bioinspired::Retina::create(orgImg.size());
		}

		// save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
		myRetina->write("RetinaDefaultParameters.xml");

		// load parameters if file exists
		myRetina->setup("RetinaSpecificParameters.xml");

		// reset all retina buffers (imagine you close your eyes for a long time)
		myRetina->clearBuffers();

		// declare retina output buffers
		cv::Mat retinaOutput_parvo;
		cv::Mat retinaOutput_magno;

		string parvoWindowName{ "Parvo" };
		namedWindow(parvoWindowName, WINDOW_NORMAL);
		moveWindow(parvoWindowName, 0, 400);

		string magnoWindowName{ "Magno" };
		namedWindow(magnoWindowName, WINDOW_NORMAL);
		moveWindow(magnoWindowName, 0, 800);

		// run retina filter on the loaded input frame
		for (auto i = 0; i < 20; i++)
			myRetina->run(orgImg);
		// Retrieve and display retina output
		myRetina->getParvo(retinaOutput_parvo);
		myRetina->getMagno(retinaOutput_magno);

		imshow(magnoWindowName, retinaOutput_magno);
		imshow(parvoWindowName, retinaOutput_parvo);
	}

	VoidAction doGammaLUT = [&]() {
		blueGamma_ = blueGammaI / 100.0;
		for (int i = 0; i < 256; ++i)
			bluep[i] = saturate_cast<uchar>(pow(i / 255.0, blueGamma_) * 255.0);
		LUT(bgr[0], bluelookUpTable, bgr[0]);
		dummy[2] = z;
		dummy[0] = bgr[0];
		merge(dummy, blueDummy);
		imshow(blueWindowName, blueDummy);

		greenGamma_ = greenGammaI / 100.0;
		for (int i = 0; i < 256; ++i)
			greenp[i] = saturate_cast<uchar>(pow(i / 255.0, greenGamma_) * 255.0);
		LUT(bgr[1], greenlookUpTable, bgr[1]);
		dummy[0] = z;
		dummy[1] = bgr[1];
		merge(dummy, greenDummy);
		imshow(greenWindowName, greenDummy);

		redGamma_ = redGammaI / 100.0;
		for (int i = 0; i < 256; ++i)
			redp[i] = saturate_cast<uchar>(pow(i / 255.0, redGamma_) * 255.0);
		LUT(bgr[2], redlookUpTable, bgr[2]);
		dummy[1] = z;
		dummy[2] = bgr[2];
		merge(dummy, redDummy);
		imshow(redWindowName, redDummy);

		Mat result;
		merge(bgr, result);
		imshow(resultWindowName, result);

		// For next round
		split(orgImg, bgr);
	};

	TrackbarCallback callbackForTrackBars = [](int pos, void* userdata)
	{
		(*(VoidAction*)userdata)();
	};

	auto intVal{ 0 };
	callbackForTrackBars(intVal, (void*)&doGammaLUT);

	createTrackbar("Gamma rot", redWindowName, &redGammaI, 200, callbackForTrackBars, (void*)&doGammaLUT);
	createTrackbar("Gamma gruen", greenWindowName, &greenGammaI, 200, callbackForTrackBars, (void*)&doGammaLUT);
	createTrackbar("Gamma blau", blueWindowName, &blueGammaI, 200, callbackForTrackBars, (void*)&doGammaLUT);

	auto wait_time = 1000;
	while (getWindowProperty(resultWindowName, WND_PROP_VISIBLE) >= 1) {
		auto keyCode = waitKey(wait_time);
		if (keyCode == 27) { // Wait for ESC key stroke
			destroyAllWindows();
			break;
		}
	}

	destroyAllWindows(); //destroy all open windows
}

void PictureCalculations::RandomForests(std::string& picName)
{
	Mat orgImg = imread(picName);
	if (orgImg.empty()) {
		cout << "no Pic selected!\n";
		return;
	}
	string orgWindowName{ "Originabild" };
	namedWindow(orgWindowName, WINDOW_NORMAL);
	moveWindow(orgWindowName, 0, 0);
	imshow(orgWindowName, orgImg);

	string sobelWindowName{ "Sobel" };
	namedWindow(sobelWindowName, WINDOW_NORMAL);
	moveWindow(sobelWindowName, 400, 0);
	
	string randomWindowName{ "Random Forest" };
	namedWindow(randomWindowName, WINDOW_NORMAL);
	moveWindow(randomWindowName, 800, 0);

	Mat orgFloat;
	orgImg.convertTo(orgFloat, DataType<float>::type, 1 / 255.0);

	Mat edges(orgFloat.size(), orgFloat.type());

	Ptr<StructuredEdgeDetection> pDollar =
		createStructuredEdgeDetection("models/model.yml");
	pDollar->detectEdges(orgFloat, edges);

	// computes orientation from edge map
	Mat orientation_map;
	pDollar->computeOrientation(edges, orientation_map);

	// suppress edges
	Mat edge_nms;
	pDollar->edgesNms(edges, orientation_map, edge_nms, 2, 0, 1, true);

	imshow(sobelWindowName, edges);
	imshow(randomWindowName, edge_nms);

	string resultWindowName{ "Original - Random Forest" };
	namedWindow(resultWindowName, WINDOW_NORMAL);
	moveWindow(resultWindowName, 400, 400);

	double minVal, maxVal;
	minMaxLoc(edge_nms, &minVal, &maxVal); //find minimum and maximum intensities
	edge_nms.convertTo(edge_nms, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

	cvtColor(edge_nms, edge_nms, COLOR_GRAY2BGR);

	edge_nms.convertTo(edge_nms, CV_8UC3);
	resize(edge_nms, edge_nms, size(orgImg));
	Mat res = orgImg - edge_nms;

	imshow(resultWindowName, res);

	// Geht wegen Lizenzbeschränkung nicht...
	//auto res = QMessageBox::question(nullptr, "Mit LSD Line Detector?",
	//	"Sollen Kanten des LSD Detectors gezeigt werden?", QMessageBox::StandardButton::Yes, QMessageBox::StandardButton::No);

	//if (res == QMessageBox::StandardButton::Yes) {
	//	cv::Mat output = orgImg.clone();

	//	/* create a random binary mask */
	//	Mat mask = Mat::ones(orgImg.size(), CV_8UC1);

	//	/* create a pointer to a BinaryDescriptor object with deafult parameters */
	//	Ptr<LSDDetector> bd = LSDDetector::createLSDDetector();

	//	/* create a structure to store extracted lines */
	//	vector<KeyLine> lines;

	//	/* extract lines */
	//	//cvtColor(orgImg, orgImg, COLOR_BGR2GRAY);
	//	orgImg = imread(picName, 1);
	//	bd->detect(orgImg, lines, 2, 1, mask);


	//	/* draw lines extracted from octave 0 */
	//	if (output.channels() == 1)
	//		cvtColor(output, output, COLOR_GRAY2BGR);
	//	for (size_t i = 0; i < lines.size(); i++)
	//	{
	//		KeyLine kl = lines[i];
	//		if (kl.octave == 0)
	//		{
	//			/* get a random color */
	//			int R = (rand() % (int)(255 + 1));
	//			int G = (rand() % (int)(255 + 1));
	//			int B = (rand() % (int)(255 + 1));

	//			/* get extremes of line */
	//			Point pt1 = Point2f(kl.startPointX, kl.startPointY);
	//			Point pt2 = Point2f(kl.endPointX, kl.endPointY);

	//			/* draw line */
	//			line(output, pt1, pt2, Scalar(B, G, R), 3);
	//		}

	//	}

	//	/* show lines on image */
	//	string lsdLines = "LSD Kanten";
	//	namedWindow(lsdLines, WINDOW_NORMAL);
	//	imshow(lsdLines, output);
	//}

	auto wait_time = 1000;
	while (getWindowProperty(randomWindowName, WND_PROP_VISIBLE) >= 1) {
		auto keyCode = waitKey(wait_time);
		if (keyCode == 27) { // Wait for ESC key stroke
			destroyAllWindows();
			break;
		}
	}
	destroyAllWindows();
}

void PictureCalculations::calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType)
{
	corners.resize(0);
	switch (patternType) {
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float(j * squareSize),
					float(i * squareSize), 0));
		break;
	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
					float(i * squareSize), 0));
		break;
	default:
		CV_Error(Error::StsBadArg, "Unknown pattern type\n");
	}
}

Mat PictureCalculations::computeHomography(const Mat& R_1to2, const Mat& tvec_1to2, const double d_inv, const Mat& normal)
{
	Mat homography = R_1to2 + d_inv * tvec_1to2 * normal.t();
	return homography;
}

void PictureCalculations::computeC2MC1(const Mat& R1, const Mat& tvec1, const Mat& R2, const Mat& tvec2, Mat& R_1to2, Mat& tvec_1to2)
{
	//c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
	R_1to2 = R2 * R1.t();
	tvec_1to2 = R2 * (-R1.t() * tvec1) + tvec2;
}

/**
 * @function randomColor
 * @brief Produces a random color given a random object
 */
Scalar PictureCalculations::randomColor(RNG& rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void PictureCalculations::decomposeHomography(const string& img1Path, const string& img2Path, const cv::Size& patternSize, const float squareSize, const string& intrinsicsPath)
{
	Mat img1 = imread(img1Path);
	Mat img2 = imread(img2Path);
	vector<Point2f> corners1, corners2;
	bool found1 = findChessboardCorners(img1, patternSize, corners1);
	bool found2 = findChessboardCorners(img2, patternSize, corners2);
	if (!found1 || !found2)
	{
		cout << "Error, cannot find the chessboard corners in both images." << endl;
		return;
	}
	vector<Point3f> objectPoints;
	calcChessboardCorners(patternSize, squareSize, objectPoints);
	FileStorage fs(intrinsicsPath, FileStorage::READ);
	Mat cameraMatrix, distCoeffs;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	vector<Point2f> objectPointsPlanar;
	for (size_t i = 0; i < objectPoints.size(); i++)
	{
		objectPointsPlanar.push_back(Point2f(objectPoints[i].x, objectPoints[i].y));
	}
	Mat rvec1, tvec1;
	solvePnP(objectPoints, corners1, cameraMatrix, distCoeffs, rvec1, tvec1);
	Mat rvec2, tvec2;
	solvePnP(objectPoints, corners2, cameraMatrix, distCoeffs, rvec2, tvec2);
	Mat R1, R2;
	Rodrigues(rvec1, R1);
	Rodrigues(rvec2, R2);
	Mat R_1to2, t_1to2;
	computeC2MC1(R1, tvec1, R2, tvec2, R_1to2, t_1to2);
	Mat rvec_1to2;
	Rodrigues(R_1to2, rvec_1to2);
	Mat normal = (Mat_<double>(3, 1) << 0, 0, 1);
	Mat normal1 = R1 * normal;
	Mat origin(3, 1, CV_64F, Scalar(0));
	Mat origin1 = R1 * origin + tvec1;
	double d_inv1 = 1.0 / normal1.dot(origin1);
	Mat homography_euclidean = computeHomography(R_1to2, t_1to2, d_inv1, normal1);
	Mat homography = cameraMatrix * homography_euclidean * cameraMatrix.inv();
	homography /= homography.at<double>(2, 2);
	homography_euclidean /= homography_euclidean.at<double>(2, 2);
	vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
	int solutions = decomposeHomographyMat(homography, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp);
	cout << "Decompose homography matrix computed from the camera displacement:" << endl << endl;
	for (int i = 0; i < solutions; i++)
	{
		double factor_d1 = 1.0 / d_inv1;
		Mat rvec_decomp;
		Rodrigues(Rs_decomp[i], rvec_decomp);
		cout << "Solution " << i << ":" << endl;
		cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
		cout << "rvec from camera displacement: " << rvec_1to2.t() << endl;
		cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << endl;
		cout << "tvec from camera displacement: " << t_1to2.t() << endl;
		cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << endl;
		cout << "plane normal at camera 1 pose: " << normal1.t() << endl << endl;
	}
	Mat H = findHomography(corners1, corners2);
	solutions = decomposeHomographyMat(H, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp);
	cout << "Decompose homography matrix estimated by findHomography():" << endl << endl;
	for (int i = 0; i < solutions; i++)
	{
		double factor_d1 = 1.0 / d_inv1;
		Mat rvec_decomp;
		Rodrigues(Rs_decomp[i], rvec_decomp);
		cout << "Solution " << i << ":" << endl;
		cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
		cout << "rvec from camera displacement: " << rvec_1to2.t() << endl;
		cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << endl;
		cout << "tvec from camera displacement: " << t_1to2.t() << endl;
		cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << endl;
		cout << "plane normal at camera 1 pose: " << normal1.t() << endl << endl;
	}

	Mat img1_warp;
	warpPerspective(img1, img1_warp, H, img1.size());	
	Mat both;
	hconcat(img1, img1_warp, both);

	string windowBoth{ "Org und homograph verschoben" };
	namedWindow(windowBoth, WINDOW_NORMAL);
	imshow(windowBoth, both);

	/// Also create a random object (RNG)
	RNG rng(0xFFFFFFFF);

	Mat img_draw_matches;
	hconcat(img1, img2, img_draw_matches);
	for (size_t i = 0; i < corners1.size(); i++)
	{
		Mat pt1 = (Mat_<double>(3, 1) << corners1[i].x, corners1[i].y, 1);
		Mat pt2 = H * pt1;
		pt2 /= pt2.at<double>(2);
		Point end((int)(img1.cols + pt2.at<double>(0)), (int)pt2.at<double>(1));
		line(img_draw_matches, corners1[i], end, randomColor(rng), 2);
	}

	string mapping{ "Zeichne Zuordnungen" };
	namedWindow(mapping, WINDOW_NORMAL);
	imshow(mapping, img_draw_matches);

	auto wait_time = 1000;
	while (getWindowProperty(windowBoth, WND_PROP_VISIBLE) >= 1) {
		auto keyCode = waitKey(wait_time);
		if (keyCode == 27) { // Wait for ESC key stroke
			destroyAllWindows();
			break;
		}
	}
	destroyAllWindows();
	cout << "Output finished!\n";
}

void PictureCalculations::Homography(string& picName)
{
	Size patternSize(9, 6);
	float squareSize = (float)0.025;
	string secondPic = QFileDialog::getOpenFileName(nullptr, "Second picture for sum", QString(), "All picture Files (*.jpg *.png *.tiff)").toStdString();
	if (secondPic.empty()) {
		cout << "nothing\n";
		destroyAllWindows();
		return;
	}
	decomposeHomography(picName,
		secondPic,
		patternSize, squareSize,
		"Pics/left_intrinsics.yml");
}

void PictureCalculations::Matches(std::string& picName)
{
	using namespace xfeatures2d;
	string secondPic = QFileDialog::getOpenFileName(nullptr, "Second picture for sum", QString(), "All picture Files (*.jpg *.png *.tiff)").toStdString();
	if (secondPic.empty()) {
		cout << "nothing\n";
		destroyAllWindows();
		return;
	}
	Mat img1 = imread(picName);
	Mat img2 = imread(secondPic);

	// detecting keypoints
	Ptr<Feature2D> surf = SURF::create();
	vector<KeyPoint> keypoints1;
	Mat descriptors1;
	surf->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
	vector<KeyPoint> keypoints2;
	Mat descriptors2;
	surf->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// drawing the results
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	string matchWindow{ "matches" };
	namedWindow(matchWindow, WINDOW_NORMAL);
	imshow(matchWindow, img_matches);
	while (waitKey(100) != 27);
	destroyAllWindows();
}
