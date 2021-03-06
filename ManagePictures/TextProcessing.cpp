#include "TextProcessing.h"

#include <vector>
#include<iostream>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <NumCpp.hpp>

using namespace cv;
using namespace std;

void TextProcessing::HandWrittenDigits()
{
	Mat gray = imread("Pics/digits.png", IMREAD_GRAYSCALE);

	vector<vector<Mat>> cells(50, vector<Mat>{});

	for (auto i = 0; i < 50; i++)
		for (auto j = 0; j < 100; j++) {
			Mat res;  
			gray(Rect(j * 20, i * 20, 20, 20)).convertTo(res, CV_32FC1);
			cells[i].emplace_back(res.reshape(1,1));
		}

	nc::NdArray<float> ncTestLabels(1, 2500);
	vector<Mat> train;
	for (auto i = 0; i < 50; i++) {
		train.insert(train.end(), cells[i].begin(), cells[i].begin() + 50) ;
	}
	vector<Mat> test;
	for (auto i = 0; i < 50; i++) {
		test.insert(test.end(), cells[i].begin() + 50, cells[i].end());
	}
	
	// Create labels for train and test data
	vector<int> train_labels, test_label;
	for (auto label = 0; label < 10; label++)
		for (auto repeat = 0; repeat < 250; repeat++) {
			train_labels.push_back(label);
			ncTestLabels.at(0, label * 250 + repeat) = (float) label;
		}
	test_label = train_labels;

	// Initiate kNN, train it on the training data, then test it with the test data with k=1
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Mat tm;
	vconcat(train, tm);
	knn->train(tm, ml::ROW_SAMPLE, train_labels);
	vconcat(test, tm);
	vector<float> result;
	Mat neighbours(size(tm), CV_32FC1), dist(size(tm), CV_32FC1);
	knn->findNearest(tm, 5, result, neighbours, dist);

	nc::NdArray<float> ncResults(result); // , ncNeighbours(neighbours), ncDist(dist);

	nc::NdArray<bool> ncMatches = ncResults == ncTestLabels;
	auto correct = nc::count_nonzero(ncMatches);
	double accuracy = correct[0] * 100.0 / ncResults.size();
	destroyAllWindows();

//    // Containers
//    nc::NdArray<int> a0 = { {1, 2}, {3, 4} };
//    nc::NdArray<int> a1 = { {1, 2}, {3, 4}, {5, 6} };
//    a1.reshape(2, 3);
//    auto a2 = a1.astype<double>();
//
//    // Initializers
//    auto a3 = nc::linspace<int>(1, 10, 5);
//    auto a4 = nc::arange<int>(3, 7);
//    auto a5 = nc::eye<int>(4);
//    auto a6 = nc::zeros<int>(3, 4);
//    auto a7 = nc::NdArray<int>(3, 4) = 0;
//    auto a8 = nc::ones<int>(3, 4);
//    auto a9 = nc::NdArray<int>(3, 4) = 1;
//    auto a10 = nc::nans(3, 4);
//    auto a11 = nc::NdArray<double>(3, 4) = nc::constants::nan;
//    auto a12 = nc::empty<int>(3, 4);
//    auto a13 = nc::NdArray<int>(3, 4);
//
//    // Slicing/Broadcasting
//    auto a14 = nc::random::randInt<int>({ 10, 10 }, 0, 100);
//    cout << a14 << "\n";
//    auto value = a14(2, 3);
//    cout << value << "\n";
//    auto slice = a14({ 2, 5 }, { 2, 5 });
//    cout << slice << "\n";
//    auto rowSlice = a14(a14.rSlice(), 7);
//    cout << rowSlice << "\n";
//    auto values = a14[a14 > 50];
//    cout << values << "\n";
//    a14.putMask(a14 > 50, 666);
//    cout << a14 << "\n";
//
//    // random
//    nc::random::seed(666);
//    auto a15 = nc::random::randN<double>({ 3, 4 });
//    cout << a15 << "\n";
//    auto a16 = nc::random::randInt<int>({ 3, 4 }, 0, 10);
//    cout << a16 << "\n";
//    auto a17 = nc::random::rand<double>({ 3, 4 });
//    cout << a17 << "\n";
//    auto a18 = nc::random::choice(a17, 3);
//    cout << a18 << "\n";
//
//    // Concatenation
//    auto a = nc::random::randInt<int>({ 3, 4 }, 0, 10);
//    auto b = nc::random::randInt<int>({ 3, 4 }, 0, 10);
//    auto c = nc::random::randInt<int>({ 3, 4 }, 0, 10);
//    cout << a << "\n";
//    cout << b << "\n";
//    cout << c << "\n";
//
//    auto a19 = nc::stack({ a, b, c }, nc::Axis::ROW);
//    cout << a19 << "\n";
//    auto a20 = nc::vstack({ a, b, c });
//    cout << a20 << "\n";
//    auto a21 = nc::hstack({ a, b, c });
//    cout << a21 << "\n";
//    auto a22 = nc::append(a, b, nc::Axis::COL);
//    cout << a22 << "\n";
//
//    // Diagonal, Traingular, and Flip
//    auto d = nc::random::randInt<int>({ 5, 5 }, 0, 10);
//    cout << d << "\n";
//    auto a23 = nc::diagonal(d);
//    cout << a23 << "\n";
//    auto a24 = nc::triu(a);
//    cout << a24 << "\n";
//    auto a25 = nc::tril(a);
//    cout << a25 << "\n";
//    auto a26 = nc::flip(d, nc::Axis::ROW);
//    cout << d << "\n";
//    cout << a26 << "\n";
//    auto a27 = nc::flipud(d);
//    cout << a27 << "\n";
//    auto a28 = nc::fliplr(d);
//    cout << a28 << "\n";
//
//    // iteration
//    for (auto it = a.begin(); it < a.end(); ++it)
//    {
//        std::cout << *it << " ";
//    }
//    std::cout << std::endl;
//
//    for (auto& arrayValue : a)
//    {
//        std::cout << arrayValue << " ";
//    }
//    std::cout << std::endl;
//
//    // Logical
//    cout << a << "\n";
//    cout << b << "\n";
//    auto a29 = nc::where(a > 5, a, b);
//    cout << a29 << "\n";
//    auto a30 = nc::any(a);
//    cout << a30 << "\n";
//    auto a31 = nc::all(a);
//    cout << a31 << "\n";
//    auto a32 = nc::logical_and(a, b);
//    cout << a32 << "\n";
//    auto a33 = nc::logical_or(a, b);
//    cout << a33 << "\n";
//    auto a34 = nc::isclose(a15, a17);
//    cout << a34 << "\n";
//    auto a35 = nc::allclose(a, b);
//    cout << a35 << "\n";
//
//    // Comparisons
//    auto a36 = nc::equal(a, b);
//    cout << a36 << "\n";
//    auto a37 = a == b;
//    cout << a37 << "\n";
//    auto a38 = nc::not_equal(a, b);
//    cout << a38 << "\n";
//    auto a39 = a != b;
//    cout << a39 << "\n";
//
//#ifdef __cpp_structured_bindings
//    auto [rows, cols] = nc::nonzero(a);
//    cout << a << "\n";
//    cout << rows << "\n";
//    cout << cols << "\n";
//#else
//    auto rowsCols = nc::nonzero(a);
//    auto& rows = rowsCols.first;
//    auto& cols = rowsCols.second;
//#endif
//
//    // Minimum, Maximum, Sorting
//    auto value1 = nc::min(a);
//    auto value2 = nc::max(a);
//    auto value3 = nc::argmin(a);
//    auto value4 = nc::argmax(a);
//    auto a41 = nc::sort(a, nc::Axis::ROW);
//    cout << a << "\n";
//    cout << a41 << "\n";
//    auto a42 = nc::argsort(a, nc::Axis::COL);
//    cout << a42 << "\n";
//    auto a43 = nc::unique(a);
//    cout << a43 << "\n";
//    auto a44 = nc::setdiff1d(a, b);
//    cout << a44 << "\n";
//    auto a45 = nc::diff(a);
//    cout << a45 << "\n";
//
//    // Reducers
//    cout << a << "\n";
//    auto value5 = nc::sum<int>(a);
//    value5 = nc::sum(a);
//    cout << value5 << "\n";
//    auto a46 = nc::sum<int>(a, nc::Axis::ROW);
//    cout << a46 << "\n";
//    auto value6 = nc::prod<int>(a);
//    cout << value6 << "\n";
//    auto a47 = nc::prod<int>(a, nc::Axis::ROW);
//    cout << a47 << "\n";
//    auto value7 = nc::mean(a);
//    cout << value7 << "\n";
//    auto a48 = nc::mean(a, nc::Axis::ROW);
//    cout << a48 << "\n";
//    auto value8 = nc::count_nonzero(a);
//    cout << value8 << "\n";
//    auto a49 = nc::count_nonzero(a, nc::Axis::ROW);
//    cout << a49 << "\n";
//
//    // I/O
//    a.print();
//    std::cout << a << std::endl;
//
//    auto tempDir = boost::filesystem::temp_directory_path();
//    auto tempTxt = (tempDir / "temp.txt").string();
//    a.tofile(tempTxt, "\n");
//    auto a50 = nc::fromfile<int>(tempTxt, "\n");
//    cout << a50 << "\n";
//
//    auto tempBin = (tempDir / "temp.bin").string();
//    nc::dump(a, tempBin);
//    auto a51 = nc::load<int>(tempBin);
//    cout << a51 << "\n";
//
//    // Mathematical Functions
//
//    // Basic Functions
//    auto a52 = nc::abs(a);
//    cout << a52 << "\n";
//    auto a53 = nc::sign(a);
//    cout << a53 << "\n";
//    auto a54 = nc::remainder(a, b);
//    cout << a54 << "\n";
//    auto a55 = nc::clip(a, 3, 8);
//    cout << a55 << "\n";
//    auto xp = nc::linspace<double>(0.0, 2.0 * nc::constants::pi, 100);
//    cout << xp << "\n";
//    auto fp = nc::sin(xp);
//    cout << fp << "\n";
//    auto x = nc::linspace<double>(0.0, 2.0 * nc::constants::pi, 1000);
//    cout << x << "\n";
//    auto f = nc::interp(x, xp, fp);
//    cout << f << "\n";
//
//    // Exponential Functions
//    auto a56 = nc::exp(a);
//    cout << a56 << "\n";
//    auto a57 = nc::expm1(a);
//    cout << a57 << "\n";
//    auto a58 = nc::log(a);
//    cout << a58 << "\n";
//    auto a59 = nc::log1p(a);
//    cout << a59 << "\n";
//
//    // Power Functions
//    auto a60 = nc::power<int>(a, 4);
//    cout << a60 << "\n";
//    auto a61 = nc::sqrt(a);
//    cout << a61 << "\n";
//    auto a62 = nc::square(a);
//    cout << a62 << "\n";
//    auto a63 = nc::cbrt(a);
//    cout << a63 << "\n";
//
//    // Trigonometric Functions
//    auto a64 = nc::sin(a);
//    cout << a64 << "\n";
//    auto a65 = nc::cos(a);
//    cout << a65 << "\n";
//    auto a66 = nc::tan(a);
//    cout << a66 << "\n";
//
//    // Hyperbolic Functions
//    auto a67 = nc::sinh(a);
//    cout << a67 << "\n";
//    auto a68 = nc::cosh(a);
//    cout << a68 << "\n";
//    auto a69 = nc::tanh(a);
//    cout << a69 << "\n";
//
//    // Classification Functions
//    auto a70 = nc::isnan(a.astype<double>());
//    cout << a70 << "\n";
//    //nc::isinf(a);
//
//    // Linear Algebra
//    auto a71 = nc::norm<int>(a);
//    cout << a71 << "\n";
//    auto a72 = nc::dot<int>(a, b.transpose());
//    cout << a72 << "\n";
//
//    auto a73 = nc::random::randInt<int>({ 3, 3 }, 0, 10);
//    cout << a73 << "\n";
//    auto a74 = nc::random::randInt<int>({ 4, 3 }, 0, 10);
//    cout << a74 << "\n";
//    auto a75 = nc::random::randInt<int>({ 1, 4 }, 0, 10);
//    cout << a75 << "\n";
//    auto value9 = nc::linalg::det(a73);
//    cout << value9 << "\n";
//    auto a76 = nc::linalg::inv(a73);
//    cout << a76 << "\n";
//    auto a77 = nc::linalg::lstsq(a74, a75);
//    cout << a77 << "\n";
//    auto a78 = nc::linalg::matrix_power<int>(a73, 3);
//    cout << a78 << "\n";
//    auto a79 = nc::linalg::multi_dot<int>({ a, b.transpose(), c });
//    cout << a79 << "\n";
//
//    nc::NdArray<double> u;
//    nc::NdArray<double> s;
//    nc::NdArray<double> vt;
//    nc::linalg::svd(a.astype<double>(), u, s, vt);
//    cout << a << "\n";
//    cout << u << "\n";
//    cout << s << "\n";
//    cout << vt << "\n";
//
//    //auto data = nc::fromfile<int>("Pics/letter-recognition.data", " ");
//    //cout << data << "\n";
}
