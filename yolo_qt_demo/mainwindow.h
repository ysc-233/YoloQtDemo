#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFileInfo>
#include <QMetaType>
#include <opencv2/opencv.hpp>
#include "yoloinference.h"

namespace Ui { class MainWindow; }

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_loadModel();
    void on_loadImage();
    void on_infer();
    void on_clear();
    void on_batchTest();
    void on_gpuCheck(int state);
    void on_imageDir();

    void handleInferenceResult(const std::vector<Detection>& detections, double elapsed_ms, const cv::Mat& result_image);

    void handleBatchProgress(int current, int total, const QString& status, const cv::Mat& result_image);

    void handleBatchFinished(int total, int success, int failed, double avg_fps);

    void handleWorkerLog(const QString& msg);

private:
    Ui::MainWindow *ui;

    QThread* workerThread;
    YOLOInference* yoloWorker;

    cv::Mat current_image;
    cv::Mat result_image;
    bool use_gpu;
    QString model_path;
    QString image_dir_path;
    QString m_currentImagePath;
    int input_width;
    int input_height;

    QStringList batch_image_paths;
    QString batch_result_dir;
    bool isBatchRunning;

    void setWidget();
    void displayImage(const cv::Mat& image);
    void initializeModel(const QString& path);
    bool loadConfigFile(const QString& model_path);
    void saveDetectionResult(const cv::Mat& image, const QString& save_path);
    void setLog(const QString log);
    void cleanupThread();
};

#endif // MAINWINDOW_H
