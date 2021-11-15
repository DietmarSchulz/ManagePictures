#include "PictureAnalyser.h"
#include <regex>
#include <algorithm>
#include <execution>
#include <TinyEXIF.h>
#include <fstream>
#include <chrono>
#include <iomanip>

//#include <exiv2/exiv2.hpp> 

#include <Windows.h>
#include <minwinbase.h>
#include <timezoneapi.h>
#include <QtCore/qstring.h>
#include <QtWidgets/qfiledialog.h>


using namespace std;
using namespace cv;

void PictureAnalyser::findIdentical(directorySet_t& pics)
{
	int numPics = (int) pics.size();
	vector<Mat> picMats(numPics);
	vector<string> picPathStrings;
	set<string> lUniquePathStrings;
	for (auto i = 0; auto & p : pics) {
		try {
			picMats[i] = imread(p.generic_string());
			if (picMats[i].data != nullptr) {
				picPathStrings.push_back(p.generic_string());
				lUniquePathStrings.insert(p.generic_string());
			}
			else {
				cout << "\nCorrupt image: " + p.generic_string() + "\n";
			}
		}
		catch (...) {
			cout << "\nException for: " << p.generic_string() << "\n";
		}
		i++;
	}

	// Remove the duplicates
	for (auto i = 0; i < numPics - 1; i++) {
		for (auto j = i + 1; j < numPics; j++) {
			if (picMats[i].size() == picMats[j].size() && picMats[i].type() == picMats[j].type()) {
				Mat difference = picMats[i] - picMats[j];
				vector<Mat> mats;
				split(difference, mats);
				bool equal{ true };
				for (auto& m : mats) {
					if (countNonZero(m) != 0) {
						equal = false;
						break;
					}
				}
				if (equal)
					lUniquePathStrings.erase(picPathStrings[j]);
			}
		}
	}
	cout << ".";
	for (auto& s : lUniquePathStrings) {
		uniquePaths.insert(s);
	}
	//int numRows = 0;
	//int numColumns = 0;
	//do {
	//	numColumns++;
	//	numRows = numPics / numColumns + ((numPics % numColumns == 0) ? 0 : 1);
	//} while (numColumns > numRows);
	//char key = waitKey();
	//destroyAllWindows();
}

#undef max
void PictureAnalyser::analyse(string_view s)
{
	if (!s.empty())
		startPath = s;
	filename2directorySet_t name2paths;

	filePathVector_t paths;
	filePathVector_t tinyPaths;
	int numPics{ 0 };
	int numTinyFiles{ 0 };
	int maxDups{ 0 };
	filesystem::recursive_directory_iterator dirs(startPath);
	copy(begin(dirs), end(dirs), std::back_inserter(paths));
	regex extReg(R"(\.jpg|\.png|\.bmp|\.tif)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);
	for (auto& p : paths) {
		if (is_regular_file(p) && regex_match(p.extension().generic_string(), extReg)) {
			name2paths[p.filename().generic_string()].insert(p);
			numPics++;
			if (filesystem::file_size(p) < 100 * 1024) {
				numTinyFiles++;
				tinyPaths.push_back(p);
			}
		}
	}

	int lookAts = 0;

	for_each(execution::par, name2paths.begin(), name2paths.end(),
		[this, &lookAts, &maxDups](auto& namePath) {
			if (auto size = namePath.second.size(); size > 1) {
				maxDups = std::max(maxDups, (int)size);
				findIdentical(namePath.second);
				lookAts++;
			}
			else {
				// only one, so the first
				uniquePaths.insert((*namePath.second.begin()).generic_string());
			}
		});

	cout << "Gesamt: " << numPics << "\n";
	cout << "Mehrfachnamen: " << lookAts << "\n";
	cout << "Max. Sequenz von Mehrfachnamen: " << maxDups << "\n";
	cout << "Winzige Bilder: " << numTinyFiles << "\n";

	// Copy unique to new directory
}

void PictureAnalyser::saveUniques()
{
	ofstream out("uniquePic.txt");
	
	for (auto& l : uniquePaths) {
		out << l << "\n";
	}
}

void PictureAnalyser::loadUniques()
{
	ifstream inp("uniquePic.txt");
	string line;

	while (getline(inp, line)) {
		uniquePaths.insert(line);
	}
}

void PictureAnalyser::copyUniques()
{
	regex oldpath(R"(c:\/pictures)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);
	for (auto& src : uniquePaths) {
		if (filesystem::file_size(src) < 100 * 1024) {
			cout << "Tiny " << src << "NOT copied!\n";
		}

		string dst = regex_replace(src, oldpath, "c:/unique_pics");
		try {
			filesystem::path destPath(dst);
			filesystem::create_directories(destPath.parent_path());
			filesystem::copy(src, dst);
		}
		catch (std::exception e) {
			cout << e.what() << "\n";
		}
		cout << dst << "\n";
	}
}

void PictureAnalyser::showDirsPicture(string& s)
{
	filePathVector_t paths;
	filesystem::recursive_directory_iterator dirs(s);
	copy(begin(dirs), end(dirs), std::back_inserter(paths));
	regex extReg(R"(\.jpg|\.png|\.bmp|\.tif)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);

	filePathVector_t imgPaths;
	for (auto& p : paths) {
		if (is_regular_file(p) && regex_match(p.extension().generic_string(), extReg)) {
			imgPaths.emplace_back(p);
		}
	}
	
	int numPics = (int)	imgPaths.size();
	int numRows = 0;
	int numColumns = 0;
	do {
		numColumns++;
		numRows = numPics / numColumns + ((numPics % numColumns == 0) ? 0 : 1);
	} while (numColumns < numRows);
	
	vector<vector<Mat>> imgs(numRows, vector<Mat>(numColumns));
	Size mSize;
	int mType;
	bool equal{ true };
	bool first{ true };
	for (auto i = 0, j = 0;  auto& s : imgPaths) {
		imgs[i][j] = imread(s.generic_string());
		if (first) {
			mSize = imgs[i][j].size();
			mType = imgs[i][j].type();
			first = false;
		}
		else if (equal && (mSize != imgs[i][j].size() || mType != imgs[i][j].type())) {
			if (mType == imgs[i][j].type()) {
				resize(imgs[i][j], imgs[i][j], mSize);
			}
			else {
				equal = false;
			}
		}
		j++;
		if (j == numColumns) {
			j = 0;
			i++;
		}
	}

	if (equal) {
		int rest = numColumns * numRows - numPics;
		if (rest > 0) {
			Mat white = Mat::zeros(mSize, mType) + Scalar(255,255,255);
			while (rest-- > 0)
				imgs[numRows - 1][numColumns - 1 - rest] = white;
		}
		vector<Mat> rowImgs(numRows);
		try {
			for (auto i = 0; auto & v : imgs) {
				hconcat(v, rowImgs[i]);
				i++;
			}
		}
		catch (Exception e) {
			cout << e.what() << "\n";
		}

		Mat all;
		try {
			vconcat(rowImgs, all);
		}
		catch (Exception e) {
			cout << e.what() << "\n";
		}
		namedWindow(s, WINDOW_NORMAL);
		imshow(s, all);
	}
	else {
		int posI = 0, posJ = 0;
		for (auto i = 0, j = 0; auto & picName : imgPaths) {
			namedWindow(s + " "s + picName.generic_string(), WINDOW_NORMAL);
			imshow(s + " "s + picName.generic_string(), imgs[i][j]);
			moveWindow(s + " "s + picName.generic_string(), posI, posJ);
			posJ += dispDelta;
			j++;
			if (j == numColumns) {
				j = 0;
				posJ = 0;
				posI += dispDelta;
				i++;
			}
		}
	}
	auto wait_time = 1000;
	if (equal) {
		while (getWindowProperty(s, WND_PROP_VISIBLE) >= 1) {
			auto keyCode = waitKey(wait_time);
			if (keyCode == 27) { // Wait for ESC key stroke
				destroyAllWindows();
				break;
			}
		}
	}
	else {
		while (waitKey(0) != 27); // Wait for ESC key stroke

		destroyAllWindows(); //destroy all open windows
	}
	cout << "Output finished!\n";
}

void PictureAnalyser::timeSortedUniques()
{
	using namespace std::literals;

	chrono::system_clock::time_point start;
	filePathVector_t paths;
	filesystem::recursive_directory_iterator dirs("c:/KameraUploads");
	copy(begin(dirs), end(dirs), std::back_inserter(paths));
	regex extReg(R"(\.jpg|\.png|\.bmp|\.tif)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);

	filePathVector_t imgPaths;
	ticks2indices_t fileTime2indices;
	for (auto i = 0;  auto & p : paths) {
		auto t = filesystem::last_write_time(p);
		SYSTEMTIME st; auto res = FileTimeToSystemTime((FILETIME*)&t, &st);
		//cout << p << "\t";
		//cout << std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count() << "\n";
		//chrono::system_clock::time_point end = start + t.time_since_epoch();
		//auto tim = chrono::system_clock::to_time_t(end);
		//tm timStr;
		//localtime_s(&timStr, &tim);
		//timStr.tm_year -= 369; // Hack!!
		//char buf[50];
		//asctime_s(buf, &timStr);
		//cout << "\t" << buf;
		//printf("\tUTC Official: Year = %d,  Month = %d,  Day = %d,  Hour = %d,  Minute = %d\n\n", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute);
		if (is_regular_file(p) && regex_match(p.extension().generic_string(), extReg)) {
			imgPaths.emplace_back(p);
			fileTime2indices[t.time_since_epoch().count()].emplace_back(i);
			i++;
		}
	}
	//if (fileTime2indices.size() != imgPaths.size())
	//	cout << "File time not unique!\n\tImagePaths: " << imgPaths.size() << "\tunique times: " << fileTime2indices.size() << "\n";
	//for (auto& tip : fileTime2indices) {
	//	if (tip.second.size() > 1) {
	//		cout << "Ticks: " << tip.first;
	//		for (auto i : tip.second) {
	//			cout << "\n\t" << imgPaths[i];
	//		}
	//		cout << "\n";
	//	}
	//}

	map<string, set<string>> picsDays;
	map<filesystem::path, filesystem::path> src2dst;

	filesystem::path dstRoot("c:/sorted_pics");

	// Store the sorted map:
	ofstream timeSortedPics("PicsSortedbyModificationTS.txt");

	// Store Copy list
	ofstream copyList("CopytoSortedPics.txt");

	for (auto& tip : fileTime2indices) {
		SYSTEMTIME st; auto res = FileTimeToSystemTime((FILETIME*)&tip.first, &st);
		stringstream dateStr;
		dateStr << st.wYear << "-" << st.wMonth << "-" << st.wDay;

		filesystem::path dstPath = dstRoot / to_string(st.wYear) / to_string(st.wMonth) / to_string(st.wDay);
		for (auto i : tip.second) {
			timeSortedPics << tip.first << ";"s;
			timeSortedPics << imgPaths[i] << ";"s;
			timeSortedPics << st.wYear << "-" << st.wMonth << "-" << st.wDay << "-" << st.wHour << ":"s << st.wMinute << ":"s << st.wSecond << "." << st.wMilliseconds << "\n";
			if (picsDays[dateStr.str()].contains(imgPaths[i].filename().generic_string())) {
				cout << "Duplicate in: " << dateStr.str() << " : " << imgPaths[i].filename().generic_string() << "\n";
				Mat m1, m2;
				m1 = imread(imgPaths[i].generic_string());
				m2 = imread((dstPath / imgPaths[i].filename().generic_string()).generic_string());
				if (m1.empty() || m2.empty()) {
					cout << "Could not read " + imgPaths[i].filename().generic_string() + "\n";
					return;
				}
				Mat difference = m1 - m2;
				vector<Mat> mats;
				split(difference, mats);
				bool equal{ true };
				for (auto& m : mats) {
					if (countNonZero(m) != 0) {
						equal = false;
						break;
					}
				}
				if (equal) {
					cout << "\tbut are identical\n";
				}
				else {
					src2dst[imgPaths[i]] = dstPath / (imgPaths[i].filename().stem().generic_string() + "_1"s + imgPaths[i].filename().extension().generic_string());
				}
			}
			else {
				picsDays[dateStr.str()].insert(imgPaths[i].filename().generic_string());
				src2dst[imgPaths[i]] = dstPath / imgPaths[i].filename().generic_string();
			}
		}
	}
	//cout << picsDays.size() << " Tage mit Bildern:\n";
	//for (auto& s : picsDays) {
	//	cout << s.first << "\n";
	//}
	for (auto& sdp : src2dst) {
		copyList << "copy " << sdp.first << " " << sdp.second << "\n";
	}

	for (auto& sdp : src2dst) {
		try {
			filesystem::create_directories(sdp.second.parent_path());
			filesystem::copy(sdp.first, sdp.second);
			cout << ".";
		}
		catch (std::exception e) {
			cout << e.what() << "\n";
		}
	}
}

void PictureAnalyser::timeSortedSDCard()
{
	filePathVector_t paths;
	filesystem::recursive_directory_iterator dirs("d:/pictures/SDCard");
	copy(begin(dirs), end(dirs), std::back_inserter(paths));
	regex extReg(R"(\.jpg|\.png|\.bmp|\.tif)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);

	filePathVector_t imgPaths;
	ticks2indices_t fileTime2indices;
	for (auto i = 0; auto & p : paths) {
		auto t = filesystem::last_write_time(p);
		SYSTEMTIME st; 
		auto res = FileTimeToSystemTime((FILETIME*)&t, &st);
		if (is_regular_file(p) && regex_match(p.extension().generic_string(), extReg)) {
			imgPaths.emplace_back(p);
			fileTime2indices[t.time_since_epoch().count()].emplace_back(i);
			i++;
		}
	}

	map<string, set<string>> picsDays;
	map<filesystem::path, filesystem::path> src2dst;

	filesystem::path dstRoot("c:/sorted_pics");

	// Store the sorted map:
	ofstream timeSortedPics("PicsSortedbyModificationTS.txt");

	// Store Copy list
	ofstream copyList("CopySDtoSortedPics.txt");

	for (auto& tip : fileTime2indices) {
		SYSTEMTIME st; auto res = FileTimeToSystemTime((FILETIME*)&tip.first, &st);
		stringstream dateStr;
		dateStr << st.wYear << "-" << st.wMonth << "-" << st.wDay;

		filesystem::path dstPath = dstRoot / to_string(st.wYear) / to_string(st.wMonth) / to_string(st.wDay);
		for (auto i : tip.second) {
			timeSortedPics << tip.first << ";"s;
			timeSortedPics << imgPaths[i] << ";"s;
			timeSortedPics << st.wYear << "-" << st.wMonth << "-" << st.wDay << "-" << st.wHour << ":"s << st.wMinute << ":"s << st.wSecond << "." << st.wMilliseconds << "\n";
			filesystem::path dstPathToFile;
			if (picsDays[dateStr.str()].contains(imgPaths[i].filename().generic_string())) {
				//cout << "Duplicate in: " << dateStr.str() << " : " << imgPaths[i].filename().generic_string() << "\n";
				dstPathToFile = dstPath / (imgPaths[i].filename().stem().generic_string() + "_1"s + imgPaths[i].filename().extension().generic_string());
			}
			else {
				dstPathToFile = dstPath / imgPaths[i].filename().generic_string();
				picsDays[dateStr.str()].insert(imgPaths[i].filename().generic_string());
			}

			auto needsCopy{ true };
			if (filesystem::exists(dstPathToFile)) {
				Mat srcImg, dstImg;
				srcImg = imread(imgPaths[i].generic_string());
				dstImg = imread(dstPathToFile.generic_string());
				if (srcImg.size() == dstImg.size() && srcImg.type() == dstImg.type()) {
					Mat difference = srcImg - dstImg;
					vector<Mat> mats;
					split(difference, mats);
					bool equal{ true };
					for (auto& m : mats) {
						if (countNonZero(m) != 0) {
							equal = false;
							break;
						}
					}
					needsCopy = !equal;
				}
				if (needsCopy) {
					cout << "Autsch: " << imgPaths[i] << "\n";
				}
				else {
					//cout << imgPaths[i] << " ist schon da!\n";
				}
			}
			else {
				cout << "new file: " << dstPathToFile.generic_string() << "\n";
			}
			if (needsCopy) {
				src2dst[imgPaths[i]] = dstPathToFile;
			}
		}
	}
	for (auto& sdp : src2dst) {
		copyList << "copy " << sdp.first << " " << sdp.second << "\n";
	}

	for (auto& sdp : src2dst) {
		try {
			filesystem::create_directories(sdp.second.parent_path());
			filesystem::copy(sdp.first, sdp.second);
			cout << ".";
		}
		catch (std::exception e) {
			cout << e.what() << "\n";
		}
	}
}

void PictureAnalyser::setDateOfFolder(string& folderPath)
{
	if (folderPath.empty()) {
		cout << "Empty path?\n";
		return;
	}
	cout << "Set " + folderPath + "'s year to? ";
	int newYear;
	cin >> newYear;

	int newMonth;
	cout << "Month?\n";
	cin >> newMonth;

	int newDay;
	cout << "Day?\n";
	cin >> newDay;

	filePathVector_t paths;
	filesystem::recursive_directory_iterator dirs(folderPath);
	copy(begin(dirs), end(dirs), std::back_inserter(paths));
	regex extReg(R"(\.jpg|\.png|\.bmp|\.tif)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);

	for (auto & p : paths) {
		if (is_regular_file(p) && regex_match(p.extension().generic_string(), extReg)) {
			auto t = filesystem::last_write_time(p);
			SYSTEMTIME st;
			auto res = FileTimeToSystemTime((FILETIME*)&t, &st);
			st.wYear = newYear;
			st.wMonth = newMonth;
			st.wDay = newDay;
			res = SystemTimeToFileTime(&st, (FILETIME*)&t);
			filesystem::last_write_time(p, t);
		}
	}
}

std::string PictureAnalyser::getPhotoTime(std::string path)
{
	//Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(argv[1]);
	//image->readMetadata();
	//Exiv2::ExifData& exifData = image->exifData();
	//if (exifData.empty()) {
	//	std::string error(argv[1]);
	//	error += ": No Exif data found in the file";
	//	throw Exiv2::Error(1, error);
	//}
	//Exiv2::ExifData::const_iterator end = exifData.end();
	//for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i) {
	//	const char* tn = i->typeName();
	//	std::cout << std::setw(44) << std::setfill(' ') << std::left
	//		<< i->key() << " "
	//		<< "0x" << std::setw(4) << std::setfill('0') << std::right
	//		<< std::hex << i->tag() << " "
	//		<< std::setw(9) << std::setfill(' ') << std::left
	//		<< (tn ? tn : "Unknown") << " "
	//		<< std::dec << std::setw(3)
	//		<< std::setfill(' ') << std::right
	//		<< i->count() << "  "
	//		<< std::dec << i->value()
	//		<< "\n";
	//}
	// open a stream to read just the necessary parts of the image file
	std::ifstream istream(path, std::ifstream::binary);

	istream.unsetf(std::ios::skipws);
	istream.seekg(0, std::ios::end);
	streamsize len = istream.tellg();
	istream.seekg(0, std::ios::beg);
	vector<unsigned char> data;
	data.reserve(len);
	data.insert(data.begin(),
		std::istream_iterator<unsigned char>(istream),
		std::istream_iterator<unsigned char>());

	// parse image EXIF and XMP metadata
	TinyEXIF::EXIFInfo imageEXIF(data.data(), len);
	if (imageEXIF.Fields)
		std::cout
			<< "Image Description " << imageEXIF.ImageDescription << "\n"
			<< "Image Resolution " << imageEXIF.ImageWidth << "x" << imageEXIF.ImageHeight << " pixels\n"
			<< "Camera Model " << imageEXIF.Make << " - " << imageEXIF.Model << "\n"
			<< "Focal Length " << imageEXIF.FocalLength << " mm\n"
			<< "Photo date " << imageEXIF.DateTimeOriginal << std::endl;
	return imageEXIF.DateTimeOriginal;
}

void PictureAnalyser::addOnlineMonth()
{
	using namespace std::literals;

	QString dummy = R"(\\fritz.box\FRITZ.NAS\Onlinespeicher\Kamera Uploads)";
	string dirToAdd = QFileDialog::getExistingDirectory(nullptr, "Open Folder", dummy).toStdString();

	filePathVector_t paths;
	filesystem::recursive_directory_iterator dirs(dirToAdd);
	copy(begin(dirs), end(dirs), std::back_inserter(paths));
	regex extReg(R"(\.jpg|\.png|\.bmp|\.tif)", wregex::flag_type::ECMAScript | wregex::flag_type::icase);
	regex dateReg(R"((\d\d\d\d):(\d\d):(\d\d)\s(\d\d):(\d\d):(\d\d))", wregex::flag_type::ECMAScript | wregex::flag_type::icase); // like 2021:07:24 17:06:22

	filePathVector_t imgPaths;
	ticks2indices_t fileTime2indices;
	for (auto i = 0; auto & p : paths) {
		auto t = filesystem::last_write_time(p);
		string photoTakenAt = getPhotoTime(p.generic_string());

		smatch m; SYSTEMTIME st;
		if (regex_match(photoTakenAt, m, dateReg)) {
			st.wYear = stoi(m[1].str());
			st.wMonth = stoi(m[2].str());
			st.wDay = stoi(m[3].str());

			st.wHour = stoi(m[4].str());
			st.wMinute = stoi(m[5].str());
			st.wSecond = stoi(m[6].str());
			st.wMilliseconds = 0;
			auto res = SystemTimeToFileTime(&st, (FILETIME*)&t);
			if (!res)
				return;
		}
		else {
			auto res = FileTimeToSystemTime((FILETIME*)&t, &st);
		}
		if (is_regular_file(p) && regex_match(p.extension().generic_string(), extReg)) {
			imgPaths.emplace_back(p);
			fileTime2indices[t.time_since_epoch().count()].emplace_back(i);
			i++;
		}
	}
	map<string, set<string>> picsDays;
	map<filesystem::path, filesystem::path> src2dst;

	filesystem::path dstRoot("c:/sorted_pics");

	// Store the sorted map:
	ofstream timeSortedPics("SinglePicsSortedbyModificationTS.txt");

	// Store Copy list
	ofstream copyList("SingleCopytoSortedPics.txt");

	for (auto& tip : fileTime2indices) {
		SYSTEMTIME st; auto res = FileTimeToSystemTime((FILETIME*)&tip.first, &st);
		stringstream dateStr;
		dateStr << st.wYear << "-" << st.wMonth << "-" << st.wDay;

		filesystem::path dstPath = dstRoot / to_string(st.wYear) / to_string(st.wMonth) / to_string(st.wDay);
		for (auto i : tip.second) {
			timeSortedPics << tip.first << ";"s;
			timeSortedPics << imgPaths[i] << ";"s;
			timeSortedPics << st.wYear << "-" << st.wMonth << "-" << st.wDay << "-" << st.wHour << ":"s << st.wMinute << ":"s << st.wSecond << "." << st.wMilliseconds << "\n";
			if (picsDays[dateStr.str()].contains(imgPaths[i].filename().generic_string())) {
				cout << "Duplicate in: " << dateStr.str() << " : " << imgPaths[i].filename().generic_string() << "\n";
				Mat m1, m2;
				m1 = imread(imgPaths[i].generic_string());
				m2 = imread((dstPath / imgPaths[i].filename().generic_string()).generic_string());
				if (m1.empty() || m2.empty()) {
					cout << "Could not read " + imgPaths[i].filename().generic_string() + "\n";
					return;
				}
				Mat difference = m1 - m2;
				vector<Mat> mats;
				split(difference, mats);
				bool equal{ true };
				for (auto& m : mats) {
					if (countNonZero(m) != 0) {
						equal = false;
						break;
					}
				}
				if (equal) {
					cout << "\tbut are identical\n";
				}
				else {
					src2dst[imgPaths[i]] = dstPath / (imgPaths[i].filename().stem().generic_string() + "_1"s + imgPaths[i].filename().extension().generic_string());
				}
			}
			else {
				picsDays[dateStr.str()].insert(imgPaths[i].filename().generic_string());
				src2dst[imgPaths[i]] = dstPath / imgPaths[i].filename().generic_string();
			}
		}
	}
	
	for (auto& sdp : src2dst) {
		copyList << "copy " << sdp.first << " " << sdp.second << "\n";
	}

	for (auto& sdp : src2dst) {
		try {
			filesystem::create_directories(sdp.second.parent_path());
			filesystem::copy(sdp.first, sdp.second);
			cout << ".";
		}
		catch (std::exception e) {
			cout << e.what() << "\n";
		}
	}
}
