#pragma once
#include <iostream>
#include <QtWidgets/qwidget.h>
#include <QtWidgets/qmessagebox.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qlistwidget.h>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qfiledialog.h>
#include "PictureAnalyser.h"
#include "PictureImprovement.h"
#include "CameraUsage.h"
#include "PictureCalculations.h"
#include "TextProcessing.h"
#include "viz3dPics.h"
#include "Tutorial.h"
#include "Croppping.h"

class ControlerWindow : public QWidget
{
	Q_OBJECT
private:
	QPushButton button{ "Quit", this };
	QPushButton buttonFile{ "Get file path", this };
	QPushButton buttonDir{ "Get directory path", this };
	QListWidget list{ this };


	std::string currFileName;
	std::string currDirName{ "d:/sorted_pics" };
	PictureAnalyser picAnalyser;
	PictureImprovement impr;
	CameraUsage cam;
	PictureCalculations calcPics;
	TextProcessing textProc;
	viz3dPics viz3d;
	Tutorial tut;
	Croppping cropping;
public:
	explicit ControlerWindow(QWidget* parent = 0) : QWidget(parent) {
		// Set size of the window
		setMinimumSize(500, 440);

		button.setToolTip("Let's quit the application!");
		button.setCheckable(true);
		button.setGeometry(10, 10, 120, 30);
		QFont font("Courier"); button.setFont(font);
		button.setIcon(QIcon::fromTheme("face-smile"));
		QObject::connect(&button, SIGNAL(clicked(bool)), this, SLOT(slotButtonClicked(bool)));

		buttonFile.setToolTip("Let's find a file path for preocessing!");
		buttonFile.setGeometry(130, 10, 120, 30);
		QObject::connect(&buttonFile, SIGNAL(clicked()), this, SLOT(slotButtonFileClicked()));

		buttonDir.setToolTip("Let's find a file path for preocessing!");
		buttonDir.setGeometry(240, 10, 120, 30);
		QObject::connect(&buttonDir, SIGNAL(clicked()), this, SLOT(slotButtonDirClicked()));

		list.setGeometry(10, 40, 480, 380);
		list.setWrapping(true);
		list.addItems(QStringList{
			"Analyse Pictures",
			"Save unique Set",
			"Load unique Set",
			"Copy unique Set",
			"Show dirs pics",
			"Simple improvement",
			"timeSortedUniques",
			"Camera objects",
			"Add pictures",
			"TimeSortedSDPics",
			"Set folder's date",
			"RGB manipulation",
			"Single hd.wr Digs",
			"Random forests",
			"Homography",
			"Matches",
			"3d Generation",
			"Add online month",
			"Tutorial: Play Around",
			"Tutorial: How to scan images",
			"Tutorial: Kernel usage",
			"Tutorial: Draw something",
			"Tutorial: Do some File IO",
			"Tutorial: Do some Filters",
			"Tutorial: Erode/Dilate",
			"Tutorial: Open/Close/MorphGrad/TopHat/BlackHat",
			"Tutorial: Zoom In/Out per pyramid",
			"Tutorial: Threshold",
			"Tutorial: Canny",
			"Tutorial: Remap",
			"Tutorial: Affine Transformation",
			"Tutorial: Histogram equalization",
			"Tutorial: Back Projection",
			"Tutorial: Match Template",
			"Tutorial: Save subpicture",
			"Tutorial: Split channel of video and save",
			"Crop",
			"Display Geometry",
			});
		QObject::connect(&list, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(listItemClicked(const QModelIndex&)));
		QObject::connect(&list, SIGNAL(clicked(const QModelIndex&)), this, SLOT(listItemClicked(const QModelIndex&)));
		button.show();
		list.show();
	};
signals:
public slots:

	void slotButtonClicked(bool checked) {
		QApplication::instance()->quit();
	}

	void slotButtonFileClicked() {
		QString dummy;
		dummy = dummy.fromStdString(currDirName);
		currFileName = QFileDialog::getOpenFileName(this, "Open Picture", dummy).toStdString();
		std::filesystem::path p = currFileName;
		currDirName = p.parent_path().string();
	}

	void slotButtonDirClicked() {
		QString dummy;
		dummy = dummy.fromStdString(currDirName);
		currDirName = QFileDialog::getExistingDirectory(this, "Open Folder", dummy).toStdString();
	}

	void listItemClicked(const QModelIndex& ix) {
		QListWidgetItem* item = list.currentItem();
		std::cout << item->text().toStdString() << " " << ix.row() << "\n";
		switch (ix.row()) {
		case 0:
			picAnalyser.analyse(currFileName);
			return;
		case 1:
			picAnalyser.saveUniques();
			return;
		case 2:
			picAnalyser.loadUniques();
			return;
		case 3:
			picAnalyser.copyUniques();
			return;
		case 4:
			picAnalyser.showDirsPicture(currDirName);
			return;
		case 5:
			impr.elementary(currFileName);
			return;
		case 6:
			picAnalyser.timeSortedUniques();
			return;
		case 7:
			cam.detectObject();
			return;
		case 8:
			calcPics.AddPicture(currFileName);
			return;
		case 9:
			picAnalyser.timeSortedSDCard();
			return;
		case 10:
			picAnalyser.setDateOfFolder(currDirName);
			return;
		case 11:
			calcPics.RGBManipulation(currFileName);
			return;
		case 12:
			textProc.HandWrittenDigits();
			return;
		case 13:
			calcPics.RandomForests(currFileName);
			return;
		case 14:
			calcPics.Homography(currFileName);
			return;
		case 15:
			calcPics.Matches(currFileName);
			return;
		case 16:
			viz3d.showPics(currFileName);
			return;
		case 17:
			picAnalyser.addOnlineMonth();
			return;
		case 18:
			tut.playAround();
			return;
		case 19:
			tut.howToScanImages(currFileName, "20", "C" /*"G"*/);
			return;
		case 20:
			tut.useKernel(currFileName);
			return;
		case 21:
			tut.drawSomething();
			return;
		case 22:
			tut.someInputOutput();
			return;
		case 23:
			tut.filters(currFileName);
			return;
		case 24:
			tut.erodeDilate(currFileName);
			return;
		case 25:
			tut.morph2(currFileName);
			return;
		case 26:
			tut.pyramid(currFileName);
			return;
		case 27:
			tut.threshold(currFileName);
			return;
		case 28:
			tut.canny(currFileName);
			return;
		case 29:
			tut.remap(currFileName);
			return;
		case 30:
			tut.affine(currFileName);
			return;
		case 31:
			tut.colorHistEqualization(currFileName);
			return;
		case 32:
			tut.backProjection(currFileName);
			return;
		case 33:
			tut.backTemplate(currFileName);
			return;
		case 34:
			tut.saveSubPicture(currFileName);
			return;
		case 35:
			tut.splitVideo();
			return;
		case 36:
			cropping.crop(currFileName);
			return;
		case 37:
			viz3d.displayGeometry();
			return;
		}
	};
};
