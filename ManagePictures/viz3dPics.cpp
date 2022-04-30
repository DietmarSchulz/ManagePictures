#include "viz3dPics.h"
#include <QtWidgets/qfiledialog.h>
#include <fstream>

using namespace cv;

void viz3dPics::showPics(std::string_view imgName)
{
	//This will be your reference camera
	viz::Viz3d myWindow("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	//Point3d min(0.25, 0.0, 0.25);
	//Point3d max(0.75, 0.5, 0.75);
	//
	//// Points bt circles
	//viz::WCube cube(min, max, false, viz::Color::blue());
	//viz::WCircle circleZ(0.05);
	//viz::WCircle circleOne(0.05, Point3d(1.0, 1.0, 1.0), Vec3d(0.0, 0.0, 1.0));

	//viz::WLine line(Point3d(1.0, 1.0, 0.0), Point3d(2.0, 2.0, 2.0), viz::Color::apricot());
	//viz::WPlane plane(Point3d(1.5, 1.5, 1.5), Vec3d(1.0, 1.0, 1.0), Vec3d(1.0, 0.0, -1.0), Size2d(1.5, 1.5), viz::Color::amethyst());
	//viz::WGrid planeGrid(Point3d(1.5, 1.5, 1.5), Vec3d(1.0, 1.0, 1.0), Vec3d(1.0, 0.0, -1.0), Vec2i::all(10), Vec2d::all(1.0), viz::Color::amethyst());

	//viz::WSphere sphere(Point3d(1.5, 1.5, 1.5), 1.0, 10, viz::Color::amethyst());

	viz::WGrid grid;

	// Aufgabe
	Point3d p0(0.0, 0.0, 0.0);
	Point3d pA(5.0, 5.0, 0.0);
	viz::WText3D tA("A", pA);
	Point3d pB(-5.0, 5.0, 0.0);
	viz::WText3D tB("B", pB);
	Point3d pC(-5.0, -5.0, 0.0);
	viz::WText3D tC("C", pC);
	Point3d pD(5.0, -5.0, 0.0);
	viz::WText3D tD("D", pD);

	Point3d pE(2.0, 0.0, 4.0);
	viz::WText3D tE("E", pE);
	Point3d pF(0.0, 2.0, 4.0);
	viz::WText3D tF("F", pF);
	Point3d pG(-2.0, 0.0, 4.0);
	viz::WText3D tG("G", pG);
	Point3d pH(0.0, -2.0, 4.0);
	viz::WText3D tH("H", pH);

	viz::WPolyLine ABCD(std::vector{ pA, pB, pC, pD, pA }, viz::Color::yellow());
	viz::WPolyLine EFGH(std::vector{ pE, pF, pG, pH, pE }, viz::Color::yellow());
	EFGH.setRenderingProperty(viz::LINE_WIDTH, 10.0);
	EFGH.setRenderingProperty(viz::OPACITY, 0.5);
	EFGH.setRenderingProperty(viz::SHADING, viz::SHADING_GOURAUD);
	viz::WPolyLine DEAFBGCHD(std::vector{ pD, pE, pA, pF, pB, pG, pC, pH, pD }, viz::Color::yellow());

	viz::WLine E0(pE, p0, viz::Color::maroon());
	viz::WLine F0(pF, p0, viz::Color::maroon());
	viz::WLine G0(pG, p0, viz::Color::maroon());
	viz::WLine H0(pH, p0, viz::Color::maroon());
	
	viz::WLine AC(pA, pC, viz::Color::black());
	viz::WLine BD(pB, pD, viz::Color::black());

	viz::WLine EG(pE, pG, viz::Color::white());
	viz::WLine FH(pF, pH, viz::Color::white());
	
	viz::WLine z(Point3d(0.0, 0.0, -5.0), Point3d(0.0, 0.0, 5.0), viz::Color::red());

	viz::WCircle circle45(8.003, Point3d(0.0, 0.0, -3.75), Vec3d(1.0, 1.0, 0.0));
	viz::WCircle circle135(8.003, Point3d(0.0, 0.0, -3.75), Vec3d(1.0, -1.0, 0.0));
	viz::WCircle circle90f(8.003, Point3d(0.0, 0.0, -3.75), Vec3d(1.0, 0.0, 0.0));
	viz::WCircle circle90(8.003, Point3d(0.0, 0.0, 0.0), Vec3d(1.0, 0.0, 0.0), 0.1, viz::Color::yellow());

	viz::WCircle circle180(8.003, Point3d(0.0, 0.0, -3.75), Vec3d(0.0, 1.0, 0.0));
	//viz::WSphere sphere(Point3d(0.0, 0.0, -3.75), 8.003, 10, viz::Color::amethyst());

	//sphere.setRenderingProperty(viz::LINE_WIDTH, 4.0);
	//myWindow.showWidget("Cube widget", cube);
	//myWindow.showWidget("Circle widget", circleZ);
	//myWindow.showWidget("CircleOne widget", circleOne);
	//myWindow.showWidget("Line widget", line);
	//myWindow.showWidget("Plane widget", plane);
	myWindow.showWidget("Grid widget", grid);
	//myWindow.showWidget("PlaneGrid widget", planeGrid);
	//myWindow.showWidget("Sphere widget", sphere);
	myWindow.showWidget("ABCD widget", ABCD);
	myWindow.showWidget("tA widget", tA);
	myWindow.showWidget("tB widget", tB);
	myWindow.showWidget("tC widget", tC);
	myWindow.showWidget("tD widget", tD);
	myWindow.showWidget("EFGH widget", EFGH);
	myWindow.showWidget("tE widget", tE);
	myWindow.showWidget("tF widget", tF);
	myWindow.showWidget("tG widget", tG);
	myWindow.showWidget("tH widget", tH);
	myWindow.showWidget("DEAFBGCHD widget", DEAFBGCHD);
	myWindow.showWidget("E0 widget", E0);
	myWindow.showWidget("F0 widget", F0);
	myWindow.showWidget("G0 widget", G0);
	myWindow.showWidget("H0 widget", H0);
	myWindow.showWidget("AC widget", AC);
	myWindow.showWidget("BD widget", BD);
	myWindow.showWidget("EG widget", EG);
	myWindow.showWidget("FH widget", FH);
	myWindow.showWidget("z widget", z);
	myWindow.showWidget("circle45 widget", circle45);
	myWindow.showWidget("circle90f widget", circle90f);
	myWindow.showWidget("circle135 widget", circle135);
	myWindow.showWidget("circle180 widget", circle180);

	myWindow.spin();

	std::vector meshPoints{ pA, pB, pC, pD, pE, pF, pG, pH };
	std::vector faces{ 4, 0, 1, 3, 2, 4, 4, 5, 7, 6, 3, 0, 4, 5, 3, 0, 1, 5, 3, 1, 5, 6, 3, 1, 2, 6, 3, 2, 6, 7, 3, 2, 3, 7, 3, 3, 7, 4, 3, 3, 4, 0 };
	viz::WMesh mesh(meshPoints, faces);
	viz::Viz3d  window("Mesh");
	mesh.setColor(viz::Color::orange_red());
	mesh.setRenderingProperty(viz::OPACITY, 0.4);
	mesh.setRenderingProperty(viz::SHADING, viz::SHADING_FLAT);
	mesh.setRenderingProperty(viz::REPRESENTATION, viz::REPRESENTATION_SURFACE);
	window.showWidget("mesh", mesh);
	window.showWidget("circle45 widget", circle45);
	window.showWidget("circle90 widget", circle90);
	circle90.setRenderingProperty(viz::LINE_WIDTH, 10);
	window.showWidget("circle135 widget", circle135);
	window.showWidget("circle180 widget", circle180);
	
	Mat rot_vecZ = Mat::zeros(1, 3, CV_32F);
	Mat rot_vecZm = Mat::zeros(1, 3, CV_32F);
	Mat rot_vecY = Mat::zeros(1, 3, CV_32F);
	
	while (!window.wasStopped())
	{
		/* Rotation using rodrigues */
		rot_vecZ.at<float>(0, 0) += (float)CV_PI * 0.0f;
		rot_vecZ.at<float>(0, 1) += (float)CV_PI * 0.0f;
		rot_vecZ.at<float>(0, 2) += (float)CV_PI * 0.01f;

		rot_vecZm.at<float>(0, 0) += (float)CV_PI * 0.0f;
		rot_vecZm.at<float>(0, 1) += (float)CV_PI * 0.0f;
		rot_vecZm.at<float>(0, 2) += -(float)CV_PI * 0.01f;

		rot_vecY.at<float>(0, 0) += (float)CV_PI * 0.0f;
		rot_vecY.at<float>(0, 1) += (float)CV_PI * 0.01f;
		rot_vecY.at<float>(0, 2) += (float)CV_PI * 0.0f;

		Mat rot_matZ;
		Mat rot_matZm;
		Mat rot_matY;
		Rodrigues(rot_vecZ, rot_matZ);
		Rodrigues(rot_vecZm, rot_matZm);
		Rodrigues(rot_vecY, rot_matY);
		Affine3f poseZ(rot_matZ);
		Affine3f poseZm(rot_matZm);
		Affine3f poseY(rot_matY, Vec3d(0.0, 0.0, -3.75));
		window.setWidgetPose("circle45 widget", poseZ);
		window.setWidgetPose("mesh", poseZm);
		window.setWidgetPose("circle90 widget", poseY);
		window.spinOnce(1, true);
	}

		// Read the camera calibration parameters
	cv::Mat cameraMatrix;
	cv::Mat cameraDistCoeffs;
	cv::FileStorage fs("calib.xml", cv::FileStorage::READ);
	fs["Intrinsic"] >> cameraMatrix;
	fs["Distortion"] >> cameraDistCoeffs;
	std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
	std::cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << std::endl;
	std::cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << std::endl;
	std::cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << std::endl << std::endl;
	cv::Matx33d cMatrix(cameraMatrix);

	// Input image points
	std::vector<cv::Point2f> imagePoints;
	imagePoints.push_back(cv::Point2f(136, 113));
	imagePoints.push_back(cv::Point2f(379, 114));
	imagePoints.push_back(cv::Point2f(379, 150));
	imagePoints.push_back(cv::Point2f(138, 135));
	imagePoints.push_back(cv::Point2f(143, 146));
	imagePoints.push_back(cv::Point2f(381, 166));
	imagePoints.push_back(cv::Point2f(345, 194));
	imagePoints.push_back(cv::Point2f(103, 161));

	// Input object points
	std::vector<cv::Point3f> objectPoints;
	objectPoints.push_back(cv::Point3f(0, 45, 0));
	objectPoints.push_back(cv::Point3f(242.5, 45, 0));
	objectPoints.push_back(cv::Point3f(242.5, 21, 0));
	objectPoints.push_back(cv::Point3f(0, 21, 0));
	objectPoints.push_back(cv::Point3f(0, 9, -9));
	objectPoints.push_back(cv::Point3f(242.5, 9, -9));
	objectPoints.push_back(cv::Point3f(242.5, 9, 44.5));
	objectPoints.push_back(cv::Point3f(0, 9, 44.5));

	// Read image
	cv::Mat image = cv::imread(imgName.data());
	if (image.empty()) {
		std::cout << "bild aussuchen!\n";
		return;
	}
	// Draw image points
	for (int i = 0; i < 8; i++) {
		cv::circle(image, imagePoints[i], 3, cv::Scalar(0, 0, 0), 2);
	}
	cv::namedWindow("An image", cv::WINDOW_NORMAL);
	cv::imshow("An image", image);

	// Create a viz window
	cv::viz::Viz3d visualizer("Viz window");
	visualizer.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	visualizer.setBackgroundColor(cv::viz::Color::white());

	/// Construct the scene
	// Create a virtual camera
	cv::viz::WCameraPosition cam(cMatrix,  // matrix of intrinsics
		image,    // image displayed on the plane
		30.0,     // scale factor
		cv::viz::Color::black());
	// Create a virtual bench from cuboids
	cv::viz::WCube plane1(cv::Point3f(0.0, 45.0, 0.0),
		cv::Point3f(242.5, 21.0, -9.0),
		true,  // show wire frame 
		cv::viz::Color::blue());
	plane1.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	cv::viz::WCube plane2(cv::Point3f(0.0, 9.0, -9.0),
		cv::Point3f(242.5, 0.0, 44.5),
		true,  // show wire frame 
		cv::viz::Color::blue());
	plane2.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	// Add the virtual objects to the environment
	visualizer.showWidget("top", plane1);
	visualizer.showWidget("bottom", plane2);
	visualizer.showWidget("Camera", cam);

	// Get the camera pose from 3D/2D points
	cv::Mat rvec, tvec;
	cv::solvePnP(objectPoints, imagePoints,      // corresponding 3D/2D pts 
		cameraMatrix, cameraDistCoeffs, // calibration 
		rvec, tvec);                    // output pose
	std::cout << " rvec: " << rvec.rows << "x" << rvec.cols << std::endl;
	std::cout << " tvec: " << tvec.rows << "x" << tvec.cols << std::endl;

	cv::Mat rotation;
	// convert vector-3 rotation
	// to a 3x3 rotation matrix
	cv::Rodrigues(rvec, rotation);

	// Move the bench	
	cv::Affine3d pose(rotation, tvec);
	visualizer.setWidgetPose("top", pose);
	visualizer.setWidgetPose("bottom", pose);

	// visualization loop
	while (cv::waitKey(100) == -1 && !visualizer.wasStopped())
	{

		visualizer.spinOnce(1,     // pause 1ms 
			true); // redraw
	}


}

void viz3dPics::displayGeometry()
{
	//This will be your reference camera
	viz::Viz3d myWindow("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	viz::WGrid gridXY;
	viz::WGrid gridXZ(Point3d(0.0, 0.0, 0.0), Vec3d(0.0, 1.0, 0.0), Vec3d(1.0, 0.0, 0.0), Vec2i::all(10), Vec2d::all(1.0), viz::Color::gray());
	viz::WGrid gridYZ(Point3d(0.0, 0.0, 0.0), Vec3d(1.0, 0.0, 0.0), Vec3d(0.0, 1.0, 0.0), Vec2i::all(10), Vec2d::all(1.0), viz::Color::olive());
	gridXY.setColor(viz::Color::black());
	myWindow.showWidget("GridXY widget", gridXY);
	myWindow.showWidget("GridXZ widget", gridXZ);
	myWindow.showWidget("GridYZ widget", gridYZ);
	myWindow.setBackgroundColor(viz::Color::Color(200.0, 200.0, 200.0));

	point_t points;
	square_t squares;
	triangle_t triangles;
	line_t lines;
	plane_t planes;

	std::string filename = QFileDialog::getOpenFileName(nullptr, "Geometry file", QString(), "All geo Files (*.geo)").toStdString();
	if (filename.empty()) {
		std::cout << "nothing\n";
		return;
	}
	std::ifstream ifs(filename);
	while (!ifs.eof()) {
		char sel;
		float x, y, z, d;
		std::string name;
		ifs >> sel;
		ifs >> name;
		switch (sel) {
		case 'p':
			ifs >> x;
			ifs >> y;
			ifs >> z;
			addPoint(points, myWindow, name, x, y, z);
			break;
		case 'l':
			{
				std::string strt;
				std::string end;
				ifs >> strt;
				ifs >> end;
				if (!points.contains(strt) || !points.contains(end))
					std::cout << "Lines points do not exist!\n";
				addLine(lines, myWindow, name, { strt, points[strt] }, { end, points[end] }); 
			}
			break;
		case 'a':
			{
				std::string strt;
				std::string end;
				ifs >> strt;
				ifs >> end;
				if (!points.contains(strt) || !points.contains(end))
					std::cout << "Lines points do not exist!\n";
				addArrow(lines, myWindow, name, { strt, points[strt] }, { end, points[end] });
			}
			break;
		case 's':
			{
				std::string a, b, c, d;
				ifs >> a;
				ifs >> b;
				ifs >> c;
				ifs >> d;
				if (!points.contains(a) || !points.contains(b) || !points.contains(c) || !points.contains(d))
					std::cout << "Square points do not exist!\n";
				addSquare(squares, myWindow, name, { a, points[a] }, { b, points[b] }, { c, points[c] }, { d, points[d] });
			}
			break;
		case 't':
			{
				std::string a, b, c;
				ifs >> a;
				ifs >> b;
				ifs >> c;
				if (!points.contains(a) || !points.contains(b) || !points.contains(c))
					std::cout << "Square points do not exist!\n";
				addTriangle(triangles, myWindow, name, { a, points[a] }, { b, points[b] }, { c, points[c] });
			}
			break;
		case 'e':
			ifs >> x;
			ifs >> y;
			ifs >> z;
			ifs >> d;
			addPlane(planes, myWindow, name, x, y, z, d);
		}
	}
	myWindow.setWindowSize(Size(2000, 2000));
	myWindow.spin();
}

void viz3dPics::addPoint(point_t& points, viz::Viz3d& window, const std::string name, float x, float y, float z)
{
	Point3f newPoint{ x, y, z };
	points[name] = newPoint;
	viz::WText3D tName(name, newPoint, 0.2);
	window.showWidget("t" + name, tName);
}

void viz3dPics::addSquare(square_t& squares, cv::viz::Viz3d& window, const std::string name, namedPoint_t a, namedPoint_t b, namedPoint_t c, namedPoint_t d)
{
	std::vector<Point3f> meshPoints{ a.second, b.second, c.second, d.second };
	squares[name] = { a, b, c, d };
	std::vector faces{ 4, 0, 1, 3, 2};
	viz::WMesh mesh(meshPoints, faces);
	mesh.setColor(viz::Color::orange_red());
	mesh.setRenderingProperty(viz::OPACITY, 0.4);
	mesh.setRenderingProperty(viz::SHADING, viz::SHADING_FLAT);
	mesh.setRenderingProperty(viz::REPRESENTATION, viz::REPRESENTATION_SURFACE);
	window.showWidget(name, mesh);
}

void viz3dPics::addTriangle(triangle_t& triangles, cv::viz::Viz3d& window, const std::string name, namedPoint_t a, namedPoint_t b, namedPoint_t c)
{
	std::vector<Point3f> meshPoints{ a.second, b.second, c.second };
	triangles[name] = { a, b, c };
	std::vector faces{ 3, 0, 1, 2 };
	viz::WMesh mesh(meshPoints, faces);
	mesh.setColor(viz::Color::azure());
	mesh.setRenderingProperty(viz::OPACITY, 0.4);
	mesh.setRenderingProperty(viz::SHADING, viz::SHADING_FLAT);
	mesh.setRenderingProperty(viz::REPRESENTATION, viz::REPRESENTATION_SURFACE);
	window.showWidget(name, mesh);

	// rectangular in a?
	Vec3f v1 = b.second - a.second;
	Vec3f v2 = c.second - a.second;
	auto n = v1.cross(v2);

	auto s = v1.dot(v2);
	if (s == 0.0) {
		std::cout << "orthogonal!\n";
		viz::WCircle rectCircle(0.1, Point3d(a.second), Vec3d(n), 0.02, viz::Color::yellow());
		window.showWidget(name + "rectAngle", rectCircle);
	}
}

void viz3dPics::addLine(line_t& lines, cv::viz::Viz3d& window, const std::string name, namedPoint_t a, namedPoint_t b)
{
	viz::WLine line(a.second, b.second);
	lines[name] = { a, b };
	line.setRenderingProperty(viz::LINE_WIDTH, 2.0);
	window.showWidget(name, line);
}

void viz3dPics::addArrow(line_t& lines, cv::viz::Viz3d& window, const std::string name, namedPoint_t a, namedPoint_t b)
{
	viz::WArrow newArrow(a.second, b.second, 0.02, viz::Color::yellow());
	lines[name] = { a, b };
	window.showWidget(name, newArrow);
}

void viz3dPics::addPlane(plane_t& planes, cv::viz::Viz3d& window, const std::string name, float a, float b, float c, float dright)
{
	float norm = dright / (a * a + b * b + c * c);
	Point3f center(a * norm, b * norm, c * norm);
	viz::WPlane newPlane(center, Vec3f(a, b, c), Vec3f(0, c, -b), Size2d(5.0, 5.0), viz::Color::amethyst());
	newPlane.setRenderingProperty(viz::OPACITY, 0.4);
	newPlane.setRenderingProperty(viz::SHADING, viz::SHADING_FLAT);
	newPlane.setRenderingProperty(viz::REPRESENTATION, viz::REPRESENTATION_SURFACE);
	viz::WArrow origin(Point3d(0.0, 0.0, 0.0), center, 0.02, viz::Color::yellow());

	window.showWidget(name, newPlane);
	window.showWidget(name + "OriginArrow", origin);

	viz::WArrow newY(center, center + Point3f(0, c, -b), 0.02, viz::Color::amethyst());
	window.showWidget(name + "NewYArrow", newY);
	viz::WArrow newX(center, center + Point3f(c, 0, -a), 0.02, viz::Color::amethyst());
	window.showWidget(name + "NewXArrow", newX);
}
