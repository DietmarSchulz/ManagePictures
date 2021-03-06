#include "PictureCalculations.h"
#include <opencv2/opencv.hpp>
#include <opencv2/bioinspired.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include <QtWidgets/qfiledialog.h>
#include <QtWidgets/qmessagebox.h>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

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
			destroyAllWindows();
			return;
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

	orgImg.convertTo(orgImg, DataType<float>::type, 1 / 255.0);

	Mat edges(orgImg.size(), orgImg.type());

	Ptr<StructuredEdgeDetection> pDollar =
		createStructuredEdgeDetection("models/model.yml");
	pDollar->detectEdges(orgImg, edges);

	// computes orientation from edge map
	Mat orientation_map;
	pDollar->computeOrientation(edges, orientation_map);

	// suppress edges
	Mat edge_nms;
	pDollar->edgesNms(edges, orientation_map, edge_nms, 2, 0, 1, true);

	imshow(sobelWindowName, edges);
	imshow(randomWindowName, edge_nms);

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
