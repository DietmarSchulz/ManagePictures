#include "Tutorial.h"
#include <chrono>
#include <QtWidgets/qfiledialog.h>
#include <filesystem>

using namespace std;
using namespace cv;

//template<Enum T>
//constexpr string to_string(T value) {
//    for (constexpr e : meta::members_of(reflexpr(T)) {
//
//    }
//}

void Tutorial::playAround()
{
	Mat M(2, 2, CV_8UC3, Scalar(0, 0, 255));
	cout << "M = " << endl << " " << M << endl << endl;

	int sz[3] = { 2,2,2 };
	Mat L(3, sz, CV_8UC(1), Scalar::all(0));
	auto ix = Vec<int, 3>(0,0,0);
	cout << "L[0, 0, 0] = " << endl << " " << L.at<int>(ix) << endl << endl;

	M.create(4, 4, CV_8UC(2));
	cout << "M = " << endl << " " << M << endl << endl;

	// Matlab style
	Mat E = Mat::eye(4, 4, CV_64F);
	cout << "E = " << endl << " " << E << endl << endl;
	Mat O = Mat::ones(2, 2, CV_32F);
	cout << "O = " << endl << " " << O << endl << endl;
	Mat Z = Mat::zeros(3, 3, CV_8UC1);
	cout << "Z = " << endl << " " << Z << endl << endl;

	// Initializer list
	Mat C = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	cout << "C = " << endl << " " << C << endl << endl;

	Mat RowClone = C.row(1).clone();
	cout << "RowClone = " << endl << " " << RowClone << endl << endl;

	// Random matrix
	Mat R = Mat(3, 2, CV_8UC3);
	randu(R, Scalar::all(0), Scalar::all(255));
	cout << "R (default) = " << endl << R << endl << endl;
	cout << "R (csv) = " << endl << cv::format(R, cv::Formatter::FormatType::FMT_CSV) << endl << endl;
	cout << "R (numpy) = " << endl << cv::format(R, cv::Formatter::FormatType::FMT_NUMPY) << endl << endl;
	cout << "R (C) = " << endl << cv::format(R, cv::Formatter::FormatType::FMT_C) << endl << endl;

	// Other formats:
	Point2f P(5, 1);
	cout << "Point (2D) = " << P << endl << endl;
	Point3f P3f(2, 6, 7);
	cout << "Point (3D) = " << P3f << endl << endl;
	vector<float> v;
	v.push_back((float)CV_PI); v.push_back(2); v.push_back(3.01f);
	cout << "Vector of floats via Mat = \n" << Mat(v) << endl << endl;
	vector<Point2f> vPoints(20);
	for (size_t i = 0; i < vPoints.size(); ++i)
		vPoints[i] = Point2f((float)(i * 5), (float)(i % 7));
	cout << "A vector of 2D Points = " << vPoints << endl << endl;
}


//! [scan-c]
Mat& Tutorial::ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    int nRows = I.rows;
    int nCols = I.cols * channels;

    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i, j;
    uchar* p;
    for (i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for (j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}
//! [scan-c]

//! [scan-iterator]
Mat& Tutorial::ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();
    switch (channels)
    {
    case 1:
    {
        MatIterator_<uchar> it, end;
        for (it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
            *it = table[*it];
        break;
    }
    case 3:
    {
        MatIterator_<Vec3b> it, end;
        for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
        {
            (*it)[0] = table[(*it)[0]];
            (*it)[1] = table[(*it)[1]];
            (*it)[2] = table[(*it)[2]];
        }
    }
    }

    return I;
}
//! [scan-iterator]

//! [scan-random]
Mat& Tutorial::ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();
    switch (channels)
    {
    case 1:
    {
        for (int i = 0; i < I.rows; ++i)
            for (int j = 0; j < I.cols; ++j)
                I.at<uchar>(i, j) = table[I.at<uchar>(i, j)];
        break;
    }
    case 3:
    {
        Mat_<Vec3b> _I = I;

        for (int i = 0; i < I.rows; ++i)
            for (int j = 0; j < I.cols; ++j)
            {
                _I(i, j)[0] = table[_I(i, j)[0]];
                _I(i, j)[1] = table[_I(i, j)[1]];
                _I(i, j)[2] = table[_I(i, j)[2]];
            }
        I = _I;
        break;
    }
    }

    return I;
}
//! [scan-random]


void Tutorial::help()
{
    std::cout
        << "\n--------------------------------------------------------------------------" << endl
        << "This program shows how to scan image objects in OpenCV (cv::Mat). As use case"
        << " we take an input image and divide the native color palette (255) with the " << endl
        << "input. Shows C operator[] method, iterators and at function for on-the-fly item address calculation." << endl
        << "Usage:" << endl
        << "./how_to_scan_images <imageNameToUse> <divideWith> [G]" << endl
        << "if you add a G parameter the image is processed in gray scale" << endl
        << "--------------------------------------------------------------------------" << endl
        << endl;
}

void Tutorial::Sharpen(const cv::Mat& myImage, cv::Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

    Result.create(myImage.size(), myImage.type());
    const int nChannels = myImage.channels();
    for (int j = 1; j < myImage.rows - 1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current = myImage.ptr<uchar>(j);
        const uchar* next = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for (int i = nChannels; i < nChannels * (myImage.cols - 1); ++i)
        {
            *output++ = saturate_cast<uchar>(5 * current[i]
                - current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
        }
    }
    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows - 1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols - 1).setTo(Scalar(0));
}

void Tutorial::MyLine(Mat img, Point start, Point end)
{
    int thickness = 2;
    int lineType = 8;
    line(img,
        start,
        end,
        Scalar(0, 0, 0),
        thickness,
        lineType);
}

void Tutorial::MyEllipse(Mat img, double angle)
{
    int thickness = 2;
    int lineType = 8;
    ellipse(img,
        Point(static_cast<int>(w / 2.0), static_cast<int>(w / 2.0)),
        Size(static_cast<int>(w / 4.0), static_cast<int>(w / 16.0)),
        angle,
        0,
        360,
        Scalar(255, 0, 0),
        thickness,
        lineType);
}

void Tutorial::MyFilledCircle(Mat img, Point center)
{
    int thickness = -1;
    int lineType = 8;
    circle(img,
        center,
        static_cast<int>(w / 32.0),
        Scalar(0, 0, 255),
        thickness,
        lineType);
}

void Tutorial::MyPolygon(cv::Mat img)
{
    int lineType = 8;
    /** Create some points */
    Point rook_points[1][20];
    rook_points[0][0] = Point(static_cast<int>(w / 4.0), static_cast<int>(7 * w / 8.0));
    rook_points[0][1] = Point(static_cast<int>(3 * w / 4.0), static_cast<int>(7 * w / 8.0));
    rook_points[0][2] = Point(static_cast<int>(3 * w / 4.0), static_cast<int>(13 * w / 16.0));
    rook_points[0][3] = Point(static_cast<int>(11 * w / 16.0), static_cast<int>(13 * w / 16.0));
    rook_points[0][4] = Point(static_cast<int>(19 * w / 32.0), static_cast<int>(3 * w / 8.0));
    rook_points[0][5] = Point(static_cast<int>(3 * w / 4.0), static_cast<int>(3 * w / 8.0));
    rook_points[0][6] = Point(static_cast<int>(3 * w / 4.0), static_cast<int>(w / 8.0));
    rook_points[0][7] = Point(static_cast<int>(26 * w / 40.0), static_cast<int>(w / 8.0));
    rook_points[0][8] = Point(static_cast<int>(26 * w / 40.0), static_cast<int>(w / 4.0));
    rook_points[0][9] = Point(static_cast<int>(22 * w / 40.0), static_cast<int>(w / 4.0));
    rook_points[0][10] = Point(static_cast<int>(22 * w / 40.0), static_cast<int>(w / 8.0));
    rook_points[0][11] = Point(static_cast<int>(18 * w / 40.0), static_cast<int>(w / 8.0));
    rook_points[0][12] = Point(static_cast<int>(18 * w / 40.0), static_cast<int>(w / 4.0));
    rook_points[0][13] = Point(static_cast<int>(14 * w / 40.0), static_cast<int>(w / 4.0));
    rook_points[0][14] = Point(static_cast<int>(14 * w / 40.0), static_cast<int>(w / 8.0));
    rook_points[0][15] = Point(static_cast<int>(w / 4.0), static_cast<int>(w / 8.0));
    rook_points[0][16] = Point(static_cast<int>(w / 4.0), static_cast<int>(3 * w / 8.0));
    rook_points[0][17] = Point(static_cast<int>(13 * w / 32.0), static_cast<int>(3 * w / 8.0));
    rook_points[0][18] = Point(static_cast<int>(5 *  w / 16.0), static_cast<int>(13 * w / 16.0));
    rook_points[0][19] = Point(static_cast<int>(w / 4.0), static_cast<int>(13 * w / 16.0));
    const Point* ppt[1] = { rook_points[0] };
    int npt[] = { 20 };
    fillPoly(img,
            ppt,
            npt,
            1,
            Scalar(255, 255, 255),
            lineType);
}

void Tutorial::howToScanImages(std::string imName, std::string intValueToReduce, std::string c)
{
    help();
    Mat I, J;
    if (c == "G")
        I = imread(imName, IMREAD_GRAYSCALE);
    else
        I = imread(imName, IMREAD_COLOR);

    if (I.empty())
    {
        cout << "The image" << imName << " could not be loaded." << endl;
        return;
    }

    namedWindow("Original", WINDOW_NORMAL);
    imshow("Original", I);

    //! [dividewith]
    int divideWith = 0; // convert our input string to number - C++ style
    stringstream s;
    s << intValueToReduce;
    s >> divideWith;
    if (!s || !divideWith)
    {
        cout << "Invalid number entered for dividing. " << endl;
        return;
    }

    uchar table[256];
    for (int i = 0; i < 256; ++i)
        table[i] = (uchar)(divideWith * (i / divideWith));
    //! [dividewith]

    const int times = 100;
    double t;

    t = (double)getTickCount();

    for (int i = 0; i < times; ++i)
    {
        cv::Mat clone_i = I.clone();
        J = ScanImageAndReduceC(clone_i, table);
    }

    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;

    cout << "Time of reducing with the C operator [] (averaged for "
        << times << " runs): " << t << " milliseconds." << endl;

    t = (double)getTickCount();

    for (int i = 0; i < times; ++i)
    {
        cv::Mat clone_i = I.clone();
        J = ScanImageAndReduceIterator(clone_i, table);
    }

    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;

    cout << "Time of reducing with the iterator (averaged for "
        << times << " runs): " << t << " milliseconds." << endl;

    t = (double)getTickCount();

    for (int i = 0; i < times; ++i)
    {
        cv::Mat clone_i = I.clone();
        ScanImageAndReduceRandomAccess(clone_i, table);
    }

    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;

    cout << "Time of reducing with the on-the-fly address generation - at function (averaged for "
        << times << " runs): " << t << " milliseconds." << endl;

    //! [table-init]
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = table[i];
    //! [table-init]

    t = (double)getTickCount();

    for (int i = 0; i < times; ++i)
        //! [table-use]
        LUT(I, lookUpTable, J);
    //! [table-use]

    t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
    t /= times;

    cout << "Time of reducing with the LUT function (averaged for "
        << times << " runs): " << t << " milliseconds." << endl;
    namedWindow("Reduced", WINDOW_NORMAL);
    imshow("Reduced", J);

    auto wait_time = 1000;
    while (getWindowProperty("Reduced", WND_PROP_VISIBLE) >= 1) {
        auto keyCode = waitKey(wait_time);
        if (keyCode == 27) { // Wait for ESC key stroke
            destroyAllWindows();
            break;
        }
    }
}

void Tutorial::useKernel(std::string imName)
{
    Mat I, J;
    I = imread(imName, IMREAD_COLOR);
    if (I.empty())
    {
        cout << "The image" << imName << " could not be loaded." << endl;
        return;
    }

    namedWindow("Original", WINDOW_NORMAL);
    imshow("Original", I);

    Sharpen(I, J);

    namedWindow("Sharpened", WINDOW_NORMAL);
    imshow("Sharpened", J);
    Mat kern = (Mat_<char>(3, 3) << 0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    Mat K;
    filter2D(I, K, I.depth(), kern);
    namedWindow("Sharpened filter", WINDOW_NORMAL);
    imshow("Sharpened filter", K);

    auto wait_time = 1000;
    while (getWindowProperty("Sharpened", WND_PROP_VISIBLE) >= 1) {
        auto keyCode = waitKey(wait_time);
        if (keyCode == 27) { // Wait for ESC key stroke
            destroyAllWindows();
            break;
        }
    }
}

void Tutorial::drawSomething()
{
    /// Windows names
    char atom_window[] = "Drawing 1: Atom";
    char rook_window[] = "Drawing 2: Rook";
    /// Create black empty images
    Mat atom_image = Mat::zeros(w, w, CV_8UC3);
    Mat rook_image = Mat::zeros(w, w, CV_8UC3);
  
    /// 1. Draw a simple atom:
    /// 1.a. Creating ellipses
    MyEllipse(atom_image, 90);
    MyEllipse(atom_image, 0);
    MyEllipse(atom_image, 45);
    MyEllipse(atom_image, -45);
    /// 1.b. Creating circles
    MyFilledCircle(atom_image, Point(static_cast<int>(w / 2.0), static_cast<int>(w / 2.0)));
    
    /// 2. Draw a rook
    /// 2.a. Create a convex polygon
    MyPolygon(rook_image);
    /// 2.b. Creating rectangles
    rectangle(rook_image,
        Point(0, static_cast<int>(7 * w / 8.0)),
        Point(w, w),
        Scalar(0, 255, 255),
        -1,
        8);
    /// 2.c. Create a few lines
    MyLine(rook_image, Point(0, 15 * w / 16), Point(w, 15 * w / 16));
    MyLine(rook_image, Point(w / 4, 7 * w / 8), Point(w / 4, w));
    MyLine(rook_image, Point(w / 2, 7 * w / 8), Point(w / 2, w));
    MyLine(rook_image, Point(3 * w / 4, 7 * w / 8), Point(3 * w / 4, w));

    namedWindow(atom_window, WINDOW_NORMAL);
    namedWindow(rook_window, WINDOW_NORMAL);
    imshow(atom_window, atom_image);
    imshow(rook_window, rook_image);
    auto wait_time = 1000;
    while (getWindowProperty(atom_window, WND_PROP_VISIBLE) >= 1) {
        auto keyCode = waitKey(wait_time);
        if (keyCode == 27) { // Wait for ESC key stroke
            destroyAllWindows();
            break;
        }
    }
}

//These write and read functions must be defined for the serialization in FileStorage to work
//! [outside]
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()) {
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}
//! [outside]

// This function will print our custom class to the console
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}

void Tutorial::someInputOutput()
{
    string filename = "outputfile.yml.gz";
    { //write
        //! [iomati]
        Mat R = Mat_<uchar>::eye(3, 3),
            T = Mat_<double>::zeros(3, 1);
        //! [iomati]
        //! [customIOi]
        MyData m(1);
        //! [customIOi]

        //! [open]
        FileStorage fs(filename, FileStorage::WRITE);
        // or:
        // FileStorage fs;
        // fs.open(filename, FileStorage::WRITE);
        //! [open]

        //! [writeNum]
        fs << "iterationNr" << 100;
        //! [writeNum]
        //! [writeStr]
        fs << "strings" << "[";                              // text - string sequence
        fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
        fs << "]";                                           // close sequence
        //! [writeStr]

        //! [writeMap]
        fs << "Mapping";                              // text - mapping
        fs << "{" << "One" << 1;
        fs << "Two" << 2 << "}";
        //! [writeMap]

        //! [iomatw]
        fs << "R" << R;                                      // cv::Mat
        fs << "T" << T;
        //! [iomatw]

        //! [customIOw]
        fs << "MyData" << m;                                // your own data structures
        //! [customIOw]

        //! [close]
        fs.release();                                       // explicit close
        //! [close]
        std::cout << "Write Done." << endl;
    }

    {//read
        std::cout << endl << "Reading: " << endl;
        FileStorage fs;
        fs.open(filename, FileStorage::READ);

        //! [readNum]
        int itNr;
        //fs["iterationNr"] >> itNr;
        itNr = (int)fs["iterationNr"];
        //! [readNum]
        std::cout << itNr;
        if (!fs.isOpened())
        {
            cerr << "Failed to open " << filename << endl;
            return;
        }

        //! [readStr]
        FileNode n = fs["strings"];                         // Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "strings is not a sequence! FAIL" << endl;
            return;
        }

        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
            std::cout << (string)*it << endl;
        //! [readStr]


        //! [readMap]
        n = fs["Mapping"];                                // Read mappings from a sequence
        std::cout << "Two  " << (int)(n["Two"]) << "; ";
        std::cout << "One  " << (int)(n["One"]) << endl << endl;
        //! [readMap]


        MyData m;
        Mat R, T;

        //! [iomat]
        fs["R"] >> R;                                      // Read cv::Mat
        fs["T"] >> T;
        //! [iomat]
        //! [customIO]
        fs["MyData"] >> m;                                 // Read your own structure_
        //! [customIO]

        std::cout << endl
            << "R = " << R << endl;
        std::cout << "T = " << T << endl << endl;
        std::cout << "MyData = " << endl << m << endl << endl;

        //Show default behavior for non existing nodes
        //! [nonexist]
        std::cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
        fs["NonExisting"] >> m;
        std::cout << endl << "NonExisting = " << endl << m << endl;
        //! [nonexist]
    }

    std::cout << endl
        << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

}

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat src; Mat dst;
char window_name[] = "Smoothing Demo";

/**
 * @function display_dst
 */
int display_dst(int delay)
{
    imshow(window_name, dst);
    int c = waitKey(delay);
    if (c >= 0) { return -1; }
    return 0;
}

/**
 * @function display_caption
 */
int display_caption(const char* caption)
{
    dst = Mat::zeros(src.size(), src.type());
    putText(dst, caption,
        Point(src.cols / 4, src.rows / 2),
        FONT_HERSHEY_COMPLEX, 4, Scalar(255, 255, 255));

    return display_dst(DELAY_CAPTION);
}


void Tutorial::filters(string& imName)
{
    namedWindow(window_name, WINDOW_NORMAL);

    /// Load the source image
    const char* filename = imName.c_str();

    src = imread(samples::findFile(filename), IMREAD_COLOR);
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Usage:\n %s [image_name-- default lena.jpg] \n", filename);
        return;
    }

    if (display_caption("Original Image") != 0)
    {
        return;
    }

    dst = src.clone();
    if (display_dst(DELAY_CAPTION) != 0)
    {
        return;
    }

    /// Applying Homogeneous blur
    if (display_caption("Homogeneous Blur") != 0)
    {
        return;
    }

    //![blur]
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        blur(src, dst, Size(i, i), Point(-1, -1));
        if (display_dst(DELAY_BLUR) != 0)
        {
            return;
        }
    }
    //![blur]

    /// Applying Gaussian blur
    if (display_caption("Gaussian Blur") != 0)
    {
        return;
    }

    //![gaussianblur]
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        GaussianBlur(src, dst, Size(i, i), 0, 0);
        if (display_dst(DELAY_BLUR) != 0)
        {
            return;
        }
    }
    //![gaussianblur]

    /// Applying Median blur
    if (display_caption("Median Blur") != 0)
    {
        return;
    }

    //![medianblur]
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        medianBlur(src, dst, i);
        if (display_dst(DELAY_BLUR) != 0)
        {
            return;
        }
    }
    //![medianblur]

    /// Applying Bilateral Filter
    if (display_caption("Bilateral Blur") != 0)
    {
        return;
    }

    //![bilateralfilter]
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        bilateralFilter(src, dst, i, i * 2, i / 2);
        if (display_dst(DELAY_BLUR) != 0)
        {
            return;
        }
    }
    //![bilateralfilter]

    /// Done
    display_caption("Done!");
}

/// Global variables
Mat erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

//![erosion]
/**
 * @function Erosion
 */
void Erosion(int, void*)
{
    int erosion_type = 0;
    if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

    //![kernel]
    Mat element = getStructuringElement(erosion_type,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size));
    //![kernel]

    /// Apply the erosion operation
    erode(src, erosion_dst, element);
    imshow("Erosion Demo", erosion_dst);
}
//![erosion]

//![dilation]
/**
 * @function Dilation
 */
void Dilation(int, void*)
{
    int dilation_type = 0;
    if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));

    /// Apply the dilation operation
    dilate(src, dilation_dst, element);
    imshow("Dilation Demo", dilation_dst);
}
//![dilation]

void Tutorial::erodeDilate(std::string& imName)
{
    src = imread(samples::findFile(imName), IMREAD_COLOR);
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << imName << " <Input image>" << endl;
        return;
    }

    /// Create windows
    namedWindow("Erosion Demo", WINDOW_NORMAL);
    namedWindow("Dilation Demo", WINDOW_NORMAL);
    //moveWindow("Dilation Demo", src.cols, 0);

    /// Create Erosion Trackbar
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
        &erosion_elem, max_elem,
        Erosion);

    createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
        &erosion_size, max_kernel_size,
        Erosion);

    /// Create Dilation Trackbar
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
        &dilation_elem, max_elem,
        Dilation);

    createTrackbar("Kernel size:\n 2n +1", "Dilation Demo",
        &dilation_size, max_kernel_size,
        Dilation);

    /// Default start
    Erosion(0, 0);
    Dilation(0, 0);

    waitKey(0);
    destroyAllWindows();
}

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
const char* window_nameMorph = "Morphology Transformations Demo";

//![morphology_operations]
/**
 * @function Morphology_Operations
 */
void Morphology_Operations(int, void*)
{
    // Since MORPH_X : 2,3,4,5 and 6
    //![operation]
    int operation = morph_operator + 2;
    //![operation]

    Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

    /// Apply the specified morphology operation
    morphologyEx(src, dst, operation, element);
    imshow(window_nameMorph, dst);
}
//![morphology_operations]

void Tutorial::morph2(std::string& imName)
{
    src = imread(imName, IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << imName << " <Input image>" << std::endl;
        return;
    }
    //![load]

    //![window]
    namedWindow(window_nameMorph, WINDOW_NORMAL); // Create window
    //![window]

    //![create_trackbar1]
    /// Create Trackbar to select Morphology operation
    createTrackbar("Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_nameMorph, &morph_operator, max_operator, Morphology_Operations);
    //![create_trackbar1]

    //![create_trackbar2]
    /// Create Trackbar to select kernel type
    createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_nameMorph,
        &morph_elem, max_elem,
        Morphology_Operations);
    //![create_trackbar2]

    //![create_trackbar3]
    /// Create Trackbar to choose kernel size
    createTrackbar("Kernel size:\n 2n +1", window_nameMorph,
        &morph_size, max_kernel_size,
        Morphology_Operations);
    //![create_trackbar3]

    /// Default start
    Morphology_Operations(0, 0);

    waitKey(0);
    destroyAllWindows();
}

const char* windowPyr_name = "Pyramids Demo";

void Tutorial::pyramid(std::string& imName)
{
    /// General instructions
    cout << "\n Zoom In-Out demo \n "
        "------------------  \n"
        " * [i] -> Zoom in   \n"
        " * [o] -> Zoom out  \n"
        " * [ESC] -> Close program \n" << endl;

    //![load]

    // Loads an image
    Mat src = imread(samples::findFile(imName));

    // Check if image is loaded fine
    if (src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default chicky_512.png] \n");
        return;
    }
    //![load]
    namedWindow(windowPyr_name, WINDOW_NORMAL);
    //![loop]
    for (;;)
    {
        //![show_image]
        imshow(windowPyr_name, src);
        //![show_image]
        char c = (char)waitKey(0);

        if (c == 27)
        {
            break;
        }
        //![pyrup]
        else if (c == 'i')
        {
            pyrUp(src, src, Size(src.cols * 2, src.rows * 2));
            printf("** Zoom In: Image x 2 \n");
        }
        //![pyrup]
        //![pyrdown]
        else if (c == 'o')
        {
            pyrDown(src, src, Size(src.cols / 2, src.rows / 2));
            printf("** Zoom Out: Image / 2 \n");
        }
        //![pyrdown]
    }
    destroyAllWindows();
    //![loop]
}

/// Global variables

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;

Mat src_gray;
const char* windowThreshold_name = "Threshold Demo";

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

//![Threshold_Demo]
/**
 * @function Threshold_Demo
 */
static void Threshold_Demo(int, void*)
{
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
    */
    threshold(src_gray, dst, threshold_value, max_binary_value, threshold_type);
    imshow(windowThreshold_name, dst);
}
//![Threshold_Demo]

void Tutorial::threshold(std::string& imName)
{
    //! [load]
    src = imread(samples::findFile(imName), IMREAD_COLOR); // Load an image

    if (src.empty())
    {
        cout << "Cannot read the image: " << imName << std::endl;
        return;
    }

    cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the image to Gray
    //! [load]

    //! [window]
    namedWindow(windowThreshold_name, WINDOW_NORMAL); // Create a window to display results
    //! [window]

    //! [trackbar]
    createTrackbar(trackbar_type,
        windowThreshold_name, &threshold_type,
        max_type, Threshold_Demo); // Create a Trackbar to choose type of Threshold

    createTrackbar(trackbar_value,
        windowThreshold_name, &threshold_value,
        max_value, Threshold_Demo); // Create a Trackbar to choose Threshold value
//! [trackbar]

    Threshold_Demo(0, 0); // Call the function to initialize

    /// Wait until the user finishes the program
    waitKey();
    destroyAllWindows();
}

//![variables]
Mat detected_edges;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratioCanny = 3;
const int kernel_size = 3;
const char* windowCanny_name = "Edge Map";
//![variables]

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void CannyThreshold(int, void*)
{
    //![reduce_noise]
    /// Reduce noise with a kernel 3x3
    blur(src_gray, detected_edges, Size(3, 3));
    //![reduce_noise]

    //![canny]
    /// Canny detector
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratioCanny, kernel_size);
    //![canny]

    /// Using Canny's output as a mask, we display our result
    //![fill]
    dst = Scalar::all(0);
    //![fill]

    //![copyto]
    src.copyTo(dst, detected_edges);
    //![copyto]

    //![display]
    imshow(windowCanny_name, dst);
    //![display]
}

void Tutorial::canny(std::string& imName)
{
    src = imread(imName, IMREAD_COLOR); // Load an image

    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << imName << " <Input image>" << std::endl;
        return;
    }
    //![load]

    //![create_mat]
    /// Create a matrix of the same type and size as src (for dst)
    dst.create(src.size(), src.type());
    //![create_mat]

    //![convert_to_gray]
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    //![convert_to_gray]

    //![create_window]
    namedWindow(windowCanny_name, WINDOW_NORMAL);
    //![create_window]

    //![create_trackbar]
    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:", windowCanny_name, &lowThreshold, max_lowThreshold, CannyThreshold);
    //![create_trackbar]

    /// Show the image
    CannyThreshold(0, 0);

    /// Wait until user exit program by pressing a key
    waitKey(0);
    destroyAllWindows();
}

/**
 * @function update_map
 * @brief Fill the map_x and map_y matrices with 4 types of mappings
 */
 //! [Update]
void update_map(int& ind, Mat& map_x, Mat& map_y)
{
    for (int i = 0; i < map_x.rows; i++)
    {
        for (int j = 0; j < map_x.cols; j++)
        {
            switch (ind)
            {
            case 0:
                if (j > map_x.cols * 0.25 && j < map_x.cols * 0.75 && i > map_x.rows * 0.25 && i < map_x.rows * 0.75)
                {
                    map_x.at<float>(i, j) = 2 * (j - map_x.cols * 0.25f) + 0.5f;
                    map_y.at<float>(i, j) = 2 * (i - map_x.rows * 0.25f) + 0.5f;
                }
                else
                {
                    map_x.at<float>(i, j) = 0;
                    map_y.at<float>(i, j) = 0;
                }
                break;
            case 1:
                map_x.at<float>(i, j) = (float)j;
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            case 2:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)i;
                break;
            case 3:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            default:
                break;
            } // end of switch
        }
    }
    ind = (ind + 1) % 4;
}
//! [Update]

void Tutorial::remap(std::string& imName)
{
    //! [Load]
    /// Load the image
    Mat src = imread(imName, IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Cannot read image: " << imName << std::endl;
        return;
    }
    //! [Load]

    //! [Create]
    /// Create dst, map_x and map_y with the same size as src:
    Mat dst(src.size(), src.type());
    Mat map_x(src.size(), CV_32FC1);
    Mat map_y(src.size(), CV_32FC1);
    //! [Create]

    //! [Window]
    /// Create window
    const char* remap_window = "Remap demo";
    namedWindow(remap_window, WINDOW_NORMAL);
    //! [Window]

    //! [Loop]
    /// Index to switch between the remap modes
    int ind = 0;
    for (;;)
    {
        /// Update map_x & map_y. Then apply remap
        update_map(ind, map_x, map_y);
        cv::remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

        /// Display results
        imshow(remap_window, dst);

        /// Each 1 sec. Press ESC to exit the program
        char c = (char)waitKey(1000);
        if (c == 27)
        {
            break;
        }
        destroyAllWindows();
    }
    //! [Loop]
}

void Tutorial::affine(std::string& imName)
{
    Mat src = imread(imName);
    if (src.empty())
    {
        return;
    }
    //! [Load the image]

    //! [Set your 3 points to calculate the  Affine Transform]
    Point2f srcTri[3];
    srcTri[0] = Point2f(0.f, 0.f);
    srcTri[1] = Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = Point2f(0.f, src.rows - 1.f);

    Point2f dstTri[3];
    dstTri[0] = Point2f(0.f, src.rows * 0.33f);
    dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
    //! [Set your 3 points to calculate the  Affine Transform]

    //! [Get the Affine Transform]
    Mat warp_mat = getAffineTransform(srcTri, dstTri);
    //! [Get the Affine Transform]

    //! [Apply the Affine Transform just found to the src image]
    /// Set the dst image the same type and size as src
    Mat warp_dst = Mat::zeros(src.rows, src.cols, src.type());

    warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    //! [Apply the Affine Transform just found to the src image]

    /** Rotating the image after Warp */

    //! [Compute a rotation matrix with respect to the center of the image]
    Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
    double angle = -50.0;
    double scale = 0.6;
    //! [Compute a rotation matrix with respect to the center of the image]

    //! [Get the rotation matrix with the specifications above]
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
    //! [Get the rotation matrix with the specifications above]

    //! [Rotate the warped image]
    Mat warp_rotate_dst;
    warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
    //! [Rotate the warped image]

    //! [Show what you got]
    namedWindow("Source image", WINDOW_NORMAL);
    namedWindow("Warp", WINDOW_NORMAL);
    namedWindow("Warp + Rotate", WINDOW_NORMAL);
    imshow("Source image", src);
    imshow("Warp", warp_dst);
    imshow("Warp + Rotate", warp_rotate_dst);
    //! [Show what you got]

    //! [Wait until user exits the program]
    waitKey();
    //! [Wait until user exits the program]
    destroyAllWindows();
}

void Tutorial::colorHistEqualization(std::string& pic)
{
    if (pic.empty())
        return;

    chrono::duration<double, std::milli> sum{ 0 };
    auto t0 = chrono::steady_clock::now();
    Mat image = imread(pic);
    //Convert the image from BGR to YCrCb color space
    Mat hist_equalized_image;
    cvtColor(image, hist_equalized_image, COLOR_BGR2YCrCb);

    //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
    vector<Mat> vec_channels;
    split(hist_equalized_image, vec_channels);

    //Equalize the histogram of only the Y channel 
    equalizeHist(vec_channels[0], vec_channels[0]);

    //Merge 3 channels in the vector to form the color image in YCrCB color space.
    merge(vec_channels, hist_equalized_image);

    //Convert the histogram equalized image from YCrCb to BGR color space again
    cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);

    //Define the names of windows
    String windowNameOfOriginalImage = "Original Image";
    String windowNameOfHistogramEqualized = "Histogram Equalized Color Image";

    // Create windows with the above names
    namedWindow(windowNameOfOriginalImage, WINDOW_NORMAL);
    namedWindow(windowNameOfHistogramEqualized, WINDOW_NORMAL);

    // Show images inside the created windows.
    imshow(windowNameOfOriginalImage, image);
    imshow(windowNameOfHistogramEqualized, hist_equalized_image);

    // Calc Histogram
    //! [Separate the image in 3 places ( B, G and R )]
    vector<Mat> bgr_planes;
    split(image, bgr_planes);
    //! [Separate the image in 3 places ( B, G and R )]

    //! [Establish the number of bins]
    int histSize = 256;
    //! [Establish the number of bins]

    //! [Set the ranges ( for B,G,R) )]
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    //! [Set the ranges ( for B,G,R) )]

    //! [Set histogram param]
    bool uniform{ true }, accumulate{ false };
    //! [Set histogram param]

    //! [Compute the histograms]
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
    //! [Compute the histograms]

    //! [Draw the histograms for B, G and R]
    int hist_w{ 512 }, hist_h{ 400 };
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    //! [Draw the histograms for B, G and R]

    //! [Normalize the result to ( 0, histImage.rows )]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    //! [Normalize the result to ( 0, histImage.rows )]

    //! [Draw for each channel]
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }
    //! [Draw for each channel]

    // Show Histogram
    namedWindow("histo", WINDOW_NORMAL);
    imshow("histo", histImage);

    auto t1 = chrono::steady_clock::now();
    sum = t1 - t0;
    cout << "Dauer: " << sum << '\n';
    //chrono::year_month_day lst = 2021y / 12 / std::chrono::last;
    //std::cout << std::format("date: {:%B %d, %Y}", lst);
    cout << chrono::system_clock::now() << '\n';
    waitKey(0); // Wait for any keystroke in any one of the windows

    destroyAllWindows(); //Destroy all opened windows
}

Mat hsv, mask;
int low = 20, up = 20;
const char* window_image = "Source image";

void Hist_and_Backproj()
{
    Mat hist;
    int h_bins = 30; int s_bins = 32;
    int histSize[] = { h_bins, s_bins };

    float h_range[] = { 0, 180 };
    float s_range[] = { 0, 256 };
    const float* ranges[] = { h_range, s_range };

    int channels[] = { 0, 1 };

    /// Get the Histogram and normalize it
    calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);

    normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

    /// Get Backprojection
    Mat backproj;
    calcBackProject(&hsv, 1, channels, hist, backproj, ranges, 1, true);

    /// Draw the backproj
    namedWindow("BackProj", WINDOW_NORMAL);
    imshow("BackProj", backproj);
}

void pickPoint(int event, int x, int y, int, void*)
{
    if (event != EVENT_LBUTTONDOWN)
    {
        return;
    }

    // Fill and get the mask
    Point seed = Point(x, y);

    int newMaskVal = 255;
    Scalar newVal = Scalar(120, 120, 120);

    int connectivity = 8;
    int flags = connectivity + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

    Mat mask2 = Mat::zeros(src.rows + 2, src.cols + 2, CV_8U);
    floodFill(src, mask2, seed, newVal, 0, Scalar(low, low, low), Scalar(up, up, up), flags);
    mask = mask2(Range(1, mask2.rows - 1), Range(1, mask2.cols - 1));

    namedWindow("Mask", WINDOW_NORMAL);
    imshow("Mask", mask);

    Hist_and_Backproj();
}

void Tutorial::backProjection(std::string& pic)
{
    if (pic.empty())
        return;
    /// Read the image
    src = imread(pic);

    /// Transform it to HSV
    cvtColor(src, hsv, COLOR_BGR2HSV);

    /// Show the image
    namedWindow(window_image, WINDOW_NORMAL);
    imshow(window_image, src);
    /// Set Trackbars for floodfill thresholds
    createTrackbar("Low thresh", window_image, &low, 255, 0);
    createTrackbar("High thresh", window_image, &up, 255, 0);
    /// Set a Mouse Callback
    setMouseCallback(window_image, pickPoint, 0);

    waitKey();
    destroyAllWindows();
}

bool use_mask;
Mat templ, result;
const char* result_window = "Result window";
int match_method;
int max_Trackbar = 5;

void MatchingMethod(int, void*)
{
    //! [copy_source]
    /// Source image to display
    Mat img_display;
    src.copyTo(img_display);
    //! [copy_source]

    //! [create_result_matrix]
    /// Create the result matrix
    int result_cols = src.cols - templ.cols + 1;
    int result_rows = src.rows - templ.rows + 1;

    result.create(result_rows, result_cols, CV_32FC1);
    //! [create_result_matrix]

    //! [match_template]
    /// Do the Matching and Normalize
    bool method_accepts_mask = (TM_SQDIFF == match_method || match_method == TM_CCORR_NORMED);
    if (use_mask && method_accepts_mask)
    {
        matchTemplate(src, templ, result, match_method, mask);
    }
    else
    {
        matchTemplate(src, templ, result, match_method);
    }
    //! [match_template]

    //! [normalize]
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    //! [normalize]

    //! [best_match]
    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    //! [best_match]

    //! [match_loc]
    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
    {
        matchLoc = minLoc;
    }
    else
    {
        matchLoc = maxLoc;
    }
    //! [match_loc]

    //! [imshow]
    /// Show me what you got
    rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
    rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

    imshow(window_image, img_display);
    imshow(result_window, result);
    //! [imshow]

    return;
}

void Tutorial::backTemplate(std::string& pic)
{
    if (pic.empty())
        return;
    /// Read the image
    src = imread(pic);

    /// Show the image
    namedWindow(window_image, WINDOW_NORMAL);
    imshow(window_image, src);
    namedWindow(result_window, WINDOW_NORMAL);

    auto roi = cv::selectROI(window_image, src);
    templ = src(roi).clone();
    string secondPic = QFileDialog::getOpenFileName(nullptr, "Second picture for comparison", QString(), "All picture Files (*.jpg *.png *.tiff)").toStdString();
    if (!secondPic.empty()) {
        src = imread(secondPic);
    }

    const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    createTrackbar(trackbar_label, window_image, &match_method, max_Trackbar, MatchingMethod);
    waitKey();
    destroyAllWindows();
}

void Tutorial::saveSubPicture(std::string& pic)
{
    if (pic.empty())
        return;
    /// Read the image
    src = imread(pic);

    /// Show the image
    namedWindow(window_image, WINDOW_NORMAL);
    imshow(window_image, src);
    namedWindow(result_window, WINDOW_NORMAL);

    auto roi = cv::selectROI(window_image, src);
    auto subPicture = src(roi).clone();
    string saveAsPath = QFileDialog::getSaveFileName(nullptr, "Save as:", QString(), "All picture Files (*.jpg *.png *.tiff)").toStdString();
    auto success = imwrite(saveAsPath, subPicture);
    if (!success) {
        cout << "Error writing the file\n";
    }
    waitKey();
    destroyAllWindows();
}

void Tutorial::splitVideo()
{
    string sourceVideo = QFileDialog::getOpenFileName(nullptr, "Video file", QString(), "All video Files (*.avi *.mpeg *.divx *.mp4)").toStdString();
    set rgbSet{ 'r', 'g', 'b' };
    set ynSet{ 'y', 'n'};
    cout << "r g b?\n";
    char channelChar, saveChar;
    do { cin >> channelChar; } while (!rgbSet.contains(channelChar));
    cout << "Ask Output type [y n]?\n";
    do { cin >> saveChar; } while (!ynSet.contains(saveChar));
    const bool askOutputType = saveChar == 'y';

    VideoCapture inputVideo(sourceVideo);              // Open input
    if (!inputVideo.isOpened())
    {
        cout << "Could not open the input video: " << sourceVideo << endl;
        return;
    }
    filesystem::path saveVideo{ sourceVideo };
    saveVideo.replace_filename(saveVideo.stem().generic_string() + channelChar + saveVideo.extension().generic_string());
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form

    // Transform from int to char via Bitwise operators
    char EXT[] = { (char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0 };

    Size S = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
        (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo;                                        // Open the output
    if (askOutputType)
        outputVideo.open(saveVideo.generic_string(), ex = -1, inputVideo.get(CAP_PROP_FPS), S, true);
    else
        outputVideo.open(saveVideo.generic_string(), ex, inputVideo.get(CAP_PROP_FPS), S, true);

    auto save{ true };
    if (!outputVideo.isOpened())
    {
        cout << "Could not open the output video for write: " << sourceVideo << endl;
        save = false;
    }

    //get the frames rate of the video
    double fps = inputVideo.get(CAP_PROP_FPS);
    cout << "Frames per seconds : " << fps << endl;
    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
        << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "Input codec type: " << EXT << endl;

    int channel = 2; // Select the channel to save
    switch (channelChar)
    {
    case 'R': channel = 2; break;
    case 'G': channel = 1; break;
    case 'B': channel = 0; break;
    }
    Mat src, res;
    vector<Mat> spl;

    String window_name_of_original_video = "Original Video";
    namedWindow(window_name_of_original_video, WINDOW_NORMAL);
    for (;;) //Show the image captured in the window and repeat
    {
        inputVideo >> src;              // read
        if (src.empty()) break;         // check if at end

        split(src, spl);                // process - extract only the correct channel
        for (int i = 0; i < 3; ++i)
            if (i != channel)
                spl[i] = Mat::zeros(S, spl[0].type());
        merge(spl, res);
        imshow(window_name_of_original_video, res);

        //outputVideo.write(res); //save or
        if (save) {
            outputVideo << res;
        }
        //wait for for 10 ms until any key is pressed.  
        //If the 'Esc' key is pressed, break the while loop.
        //If the any other key is pressed, continue the loop 
        //If any key is not pressed withing 10 ms, continue the loop
        if (waitKey(static_cast<int>(1000 / fps)) == 27)
        {
            cout << "Esc key is pressed by user. Stoppig the video" << endl;
            break;
        }
    }
    if (save) {
        cout << "Finished writing" << endl;
    }
    else {
        cout << "Video finished\n";
    }
    destroyAllWindows();
}
