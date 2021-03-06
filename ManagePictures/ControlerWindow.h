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

class ControlerWindow : public QWidget
{
	Q_OBJECT
private:
	QPushButton button{ "Quit", this };
	QPushButton buttonFile{ "Get file path", this };
	QPushButton buttonDir{ "Get directory path", this };
	QListWidget list{ this };


	std::string currFileName;
	std::string currDirName{ "C:/sorted_pics" };
	PictureAnalyser picAnalyser;
	PictureImprovement impr;
	CameraUsage cam;
	PictureCalculations calcPics;
	TextProcessing textProc;
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
			"Set folder's year",
			"RGB manipulation",
			"Single hd.wr Digs",
			"Random forests",
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
			picAnalyser.setYearOfFolder(currDirName);
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
		}
	};
};
