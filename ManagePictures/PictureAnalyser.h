#pragma once

#include <filesystem>
#include <string>
#include <map>
#include <unordered_set>

#include <opencv2/opencv.hpp>

class PictureAnalyser
{
	using directorySet_t = std::set<std::filesystem::path>;
	using filename2directorySet_t = std::map<std::string, directorySet_t>;
	using filePathVector_t = std::vector<std::filesystem::path>;
	using ticks2indices_t = std::map<long long, std::vector<int>>;
	const int dispDelta{300};

	std::filesystem::path startPath{"d:/Pictures"};
	std::set<std::string> uniquePaths;
	void findIdentical(directorySet_t& pics);
	std::string getPhotoTime(std::string path);
public:
	void analyse(std::string_view s);
	void saveUniques();
	void loadUniques();
	void copyUniques();
	void showDirsPicture(std::string& s);
	void timeSortedUniques();
	void timeSortedSDCard();
	void setDateOfFolder(std::string& folderPath);
	void addOnlineMonth();
};

