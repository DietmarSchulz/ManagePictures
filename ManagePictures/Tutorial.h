#pragma once

#include <opencv2/opencv.hpp>

class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
    {}
    //! [inside]
    void write(cv::FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
    void read(const cv::FileNode& node)                          //Read serialization for this class
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (std::string)node["id"];
    }
    //! [inside]
public:   // Data Members
    int A;
    double X;
    std::string id;
};

class Tutorial
{
private:
	int w = 400;

	static cv::Mat& ScanImageAndReduceC(cv::Mat& I, const uchar* const table);
	static cv::Mat& ScanImageAndReduceIterator(cv::Mat& I, const uchar* const table);
	static cv::Mat& ScanImageAndReduceRandomAccess(cv::Mat& I, const uchar* const table);
	static void help();
	static void Sharpen(const cv::Mat& myImage, cv::Mat& Result);
	void MyLine(cv::Mat img, cv::Point start, cv::Point end);
	void MyEllipse(cv::Mat img, double angle);
	void MyFilledCircle(cv::Mat img, cv::Point center);
	void MyPolygon(cv::Mat img);
public:
	void playAround();
	void howToScanImages(std::string imName, std::string intValueToReduce, std::string c);
	void useKernel(std::string imName);
	void drawSomething();
	void someInputOutput();
    void filters(std::string& imName);
    void erodeDilate(std::string& imName);
    void morph2(std::string& imName);
    void pyramid(std::string& imName);
    void threshold(std::string& imName);
    void canny(std::string& imName);
    void remap(std::string& imName);
    void affine(std::string& imName);
    void colorHistEqualization(std::string& pic);
    void backProjection(std::string& pic);
    void backTemplate(std::string& pic);
    void saveSubPicture(std::string& pic);
};

