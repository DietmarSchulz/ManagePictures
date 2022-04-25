#pragma once

#include <string>
#include <map>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

class viz3dPics
{
public:
	using point_t = std::map<std::string, cv::Point3f>;
	using namedPoint_t = std::pair<std::string, cv::Point3f>;
	using square_t = std::map<std::string, std::array<namedPoint_t, 4>>;
	using line_t = std::map<std::string, std::array<namedPoint_t, 2>>;

	void showPics(std::string_view imgName);
	void displayGeometry();
private:
	void addPoint(point_t& points, cv::viz::Viz3d& window, const std::string name, float x, float y, float z);
	void addSquare(square_t& squares, cv::viz::Viz3d& window, const std::string name, namedPoint_t a, namedPoint_t b, namedPoint_t c, namedPoint_t d);
	void addLine(line_t& lines, cv::viz::Viz3d& window, const std::string name, namedPoint_t a, namedPoint_t b);
};

