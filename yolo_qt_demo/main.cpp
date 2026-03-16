#include "mainwindow.h"

#include <QApplication>
#include <QMetaType>
#include <opencv2/opencv.hpp>
#include "yoloinference.h"

int main(int argc, char *argv[])
{
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::vector<Detection>>("std::vector<Detection>");
    qRegisterMetaType<QStringList>("QStringList");

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
