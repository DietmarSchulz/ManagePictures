#include <iostream>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qpushbutton.h>
#include "opencv2/opencv.hpp"

#include "ControlerWindow.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	ControlerWindow mwindow;
	mwindow.show();

	app.exec();
}
