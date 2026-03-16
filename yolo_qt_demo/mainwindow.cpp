#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QCoreApplication>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QMetaObject>
#include <QTimer>
#include <iostream>
#include <chrono>
#include <QDebug>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      workerThread(nullptr),
      yoloWorker(nullptr),
      use_gpu(true), input_width(640), input_height(640),
      isBatchRunning(false)
{
    ui->setupUi(this);
    setWidget();
}

MainWindow::~MainWindow()
{
    cleanupThread();
    delete ui;
}

void MainWindow::cleanupThread()
{
    if (workerThread) {
        if (workerThread->isRunning()) {
            workerThread->quit();
            workerThread->wait(3000);
        }
        delete workerThread;
        workerThread = nullptr;
    }
    if (yoloWorker) {
        yoloWorker->deleteLater();
        yoloWorker = nullptr;
    }
}

void MainWindow::initializeModel(const QString& path)
{
    qDebug() << __FUNCTION__ << "Start..";

    cleanupThread();

    QString model_path_to_load = path;
    ui->ckb_useGPU->setChecked(use_gpu);

    if (model_path_to_load.isEmpty()) {
        QString dirPath = QCoreApplication::applicationDirPath();
        model_path_to_load = dirPath + "/models/test.onnx";
    }

    QFileInfo file_info(model_path_to_load);
    if (!file_info.exists()) {
        if (!path.isEmpty()) {
            QMessageBox::critical(this, "Critical Error", "Model file not found!\n\nPath: " + model_path_to_load);
        }
        setLog("Status: Model Not Loaded");
        ui->btn_infer->setEnabled(false);
        ui->btn_Batch->setEnabled(false);
        return;
    }

    loadConfigFile(model_path_to_load);

    workerThread = new QThread(this);
    yoloWorker = new YOLOInference(
        model_path_to_load.toStdString(),
        use_gpu,
        0.25f, 0.45f,
        input_width, input_height
    );

    yoloWorker->moveToThread(workerThread);

    connect(yoloWorker, &YOLOInference::logSignal, this, &MainWindow::handleWorkerLog, Qt::QueuedConnection);
    connect(yoloWorker, &YOLOInference::inferenceFinished, this, &MainWindow::handleInferenceResult, Qt::QueuedConnection);
    connect(yoloWorker, &YOLOInference::batchProgress, this, &MainWindow::handleBatchProgress, Qt::QueuedConnection);
    connect(yoloWorker, &YOLOInference::batchFinished, this, &MainWindow::handleBatchFinished, Qt::QueuedConnection);

    workerThread->start();

    if (!yoloWorker->isLoaded()) {
        if (use_gpu) {
            std::cout << "GPU initialization failed, trying CPU..." << std::endl;
            use_gpu = false;
            ui->ckb_useGPU->setChecked(false);
            cleanupThread();
            workerThread = new QThread(this);
            yoloWorker = new YOLOInference(
                model_path_to_load.toStdString(),
                false,
                0.25f, 0.45f,
                input_width, input_height
            );
            yoloWorker->moveToThread(workerThread);

            connect(yoloWorker, &YOLOInference::logSignal, this, &MainWindow::handleWorkerLog, Qt::QueuedConnection);
            connect(yoloWorker, &YOLOInference::inferenceFinished, this, &MainWindow::handleInferenceResult, Qt::QueuedConnection);
            connect(yoloWorker, &YOLOInference::batchProgress, this, &MainWindow::handleBatchProgress, Qt::QueuedConnection);
            connect(yoloWorker, &YOLOInference::batchFinished, this, &MainWindow::handleBatchFinished, Qt::QueuedConnection);

            workerThread->start();

            if (!yoloWorker->isLoaded()) {
                QMessageBox::critical(this, "Critical Error",
                    "Failed to load model!\n\nPossible reasons:\n"
                    "1. Invalid ONNX file format\n"
                    "2. OpenCV version incompatibility\n"
                    "3. Corrupted model file");
                setLog("Status: Model Load Failed");
                ui->btn_infer->setEnabled(false);
                ui->btn_Batch->setEnabled(false);
                ui->ckb_useGPU->setChecked(false);
                return;
            }
        } else {
            QMessageBox::critical(this, "Critical Error", "Failed to load model!");
            setLog("Status: Model Load Failed");
            ui->btn_infer->setEnabled(false);
            ui->btn_Batch->setEnabled(false);
            return;
        }
    }

    this->model_path = model_path_to_load;
    setLog(QString("Status: Ready (%1, %2x%3) ")
           .arg(use_gpu ? "GPU" : "CPU")
           .arg(input_width).arg(input_height));

    ui->btn_infer->setEnabled(!current_image.empty());
    if(!batch_image_paths.empty())
        ui->btn_Batch->setEnabled(true);
}

void MainWindow::setWidget()
{
    qDebug() << __FUNCTION__ << "Start..";
    setLog("Status: Initializing...");
    setWindowTitle("YOLO Object Detection Demo");
    ui->lbl_image->setStyleSheet("QLabel { background-color: #2b2b2b; color: white; border: 1px solid #555; }");
    ui->lbl_image->setText("Please load an image");

    connect(ui->btn_loadModule, &QPushButton::clicked, this, &MainWindow::on_loadModel);
    connect(ui->btn_loadImage, &QPushButton::clicked, this, &MainWindow::on_loadImage);
    connect(ui->btn_setImageDir, &QPushButton::clicked, this, &MainWindow::on_imageDir);
    connect(ui->btn_infer, &QPushButton::clicked, this, &MainWindow::on_infer);
    connect(ui->btn_Batch, &QPushButton::clicked, this, &MainWindow::on_batchTest);
    connect(ui->btn_Clear, &QPushButton::clicked, this, &MainWindow::on_clear);
    connect(ui->ckb_useGPU, &QCheckBox::stateChanged, this, &MainWindow::on_gpuCheck);
    ui->btn_infer->setEnabled(false);
    ui->btn_Batch->setEnabled(false);
    ui->ckb_useGPU->setChecked(true);
}

void MainWindow::on_loadModel()
{
    QString file_path = QFileDialog::getOpenFileName(
        this, "Select ONNX Model", "",
        "ONNX Files (*.onnx)"
    );
    if (file_path.isEmpty()) {
        return;
    }
    setLog("Status: Loading Model...");
    initializeModel(file_path);
    if (yoloWorker->isLoaded()) {
        setLog("Warming up GPU...");
        cv::Mat dummy_image(input_height, input_width, CV_8UC3, cv::Scalar(0, 0, 0));
        QMetaObject::invokeMethod(yoloWorker, "processImage", Qt::QueuedConnection,
                                  Q_ARG(cv::Mat, dummy_image.clone()));
        setLog("GPU Warm-up complete.");
    }
}

void MainWindow::on_loadImage()
{
    QString file_path = QFileDialog::getOpenFileName(
        this, "Select Image", "",
        "Image Files (*.jpg *.jpeg *.png *.bmp *.tif)"
    );
    if (file_path.isEmpty()) {
        return;
    }

    m_currentImagePath = file_path;
    current_image = cv::imread(file_path.toStdString());

    if (current_image.empty()) {
        QMessageBox::warning(this, "Error", "Failed to load image!");
        return;
    }

    result_image = current_image.clone();
    displayImage(result_image);

    ui->btn_infer->setEnabled(yoloWorker && yoloWorker->isLoaded());
    setLog(QString("Status: Image Loaded (%1 x %2)")
           .arg(current_image.cols).arg(current_image.rows));
}

void MainWindow::on_imageDir()
{
    QString dir_path = QFileDialog::getExistingDirectory(
        this, "Select Image Directory", ""
    );
    if (dir_path.isEmpty()) {
        return;
    }

    image_dir_path = dir_path;
    if(yoloWorker->isLoaded())
        ui->btn_Batch->setEnabled(true);
    setLog(QString("Status: Image Dir Set (%1)").arg(dir_path));
}

void MainWindow::on_infer()
{
    if (current_image.empty() || !yoloWorker || !yoloWorker->isLoaded()) {
        QMessageBox::warning(this, "Warning", "Please load an image or check the model!");
        return;
    }

    if (isBatchRunning) {
        QMessageBox::warning(this, "Warning", "Batch test is running, please wait!");
        return;
    }

    setLog("Status: Running Inference...");
    ui->btn_infer->setEnabled(false);
    ui->lbl_image->setText("Processing...");

    QMetaObject::invokeMethod(yoloWorker, "processImage", Qt::QueuedConnection,
                              Q_ARG(cv::Mat, current_image.clone()));
}

void MainWindow::handleInferenceResult(const std::vector<Detection>& detections, double elapsed_ms, const cv::Mat& result_image)
{

    ui->btn_infer->setEnabled(true);

    if (current_image.empty()) return;

    float fps = (elapsed_ms > 0) ? (1000.0f / elapsed_ms) : 0.0f;

    this->result_image = result_image.clone();
    displayImage(this->result_image);

    QString dirPath = QCoreApplication::applicationDirPath();
    QString save_path = dirPath + "/detRes";
    QDir().mkpath(save_path);

    QFileInfo fi(m_currentImagePath);
    QString base_name = fi.completeBaseName();
    QString save_file = save_path + "/" + base_name + "_det.jpg";
    saveDetectionResult(this->result_image, save_file);

    setLog(QString("Status: %1 object(s) detected").arg(detections.size()));
    setLog(QString("Inference Time: %1 ms | FPS: %2").arg(elapsed_ms, 0, 'f', 2).arg(fps, 0, 'f', 1));

    for (const auto & det : detections) {
        qDebug() << QString("Class: %1 | Confidence: %2 | Box: [%3, %4, %5, %6]")
                    .arg(det.class_name.c_str())
                    .arg(det.confidence, 0, 'f', 2)
                    .arg(det.bbox.x).arg(det.bbox.y)
                    .arg(det.bbox.width).arg(det.bbox.height);
    }
}

void MainWindow::on_batchTest()
{
    if (image_dir_path.isEmpty() || !yoloWorker || !yoloWorker->isLoaded()) {
        QMessageBox::warning(this, "Warning",
            "Please set image directory and load model first!");
        return;
    }

    if (isBatchRunning) {
        QMessageBox::warning(this, "Warning", "Batch test is already running!");
        return;
    }

    QDir dir(image_dir_path);
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.tif";
    dir.setNameFilters(filters);

    QFileInfoList file_list = dir.entryInfoList(QDir::Files);

    if (file_list.isEmpty()) {
        QMessageBox::warning(this, "Warning", "No images found in directory!");
        return;
    }

    batch_image_paths.clear();
    for (const auto& fi : file_list) {
        batch_image_paths << fi.filePath();
    }

    batch_result_dir = image_dir_path + "/detRes";
    isBatchRunning = true;

    ui->btn_Batch->setEnabled(false);
    ui->btn_infer->setEnabled(false);

    setLog(QString("Status: Batch Testing 0/%1...").arg(batch_image_paths.size()));

    QMetaObject::invokeMethod(yoloWorker, "processBatch", Qt::QueuedConnection,
                              Q_ARG(QStringList, batch_image_paths),
                              Q_ARG(QString, batch_result_dir));
}

void MainWindow::handleBatchProgress(int current, int total, const QString& status , const cv::Mat& result_image)
{
    this->result_image = result_image.clone();
    displayImage(this->result_image);
    setLog(QString("Status: Batch Testing %1/%2... %3").arg(current).arg(total).arg(status));
}

void MainWindow::handleBatchFinished(int total, int success, int failed, double avg_fps)
{
    isBatchRunning = false;

    ui->btn_Batch->setEnabled(true);
    ui->btn_infer->setEnabled(!current_image.empty());

    QMessageBox::information(this, "Batch Test Complete",
        QString("Total: %1\nSuccess: %2\nFailed: %3\nAvg FPS: %4")
            .arg(total).arg(success).arg(failed).arg(avg_fps, 0, 'f', 1));

    setLog(QString("Status: Batch Complete (%1/%2)").arg(success).arg(total));
    setLog(QString("Avg FPS: %1").arg(avg_fps, 0, 'f', 1));
}

void MainWindow::handleWorkerLog(const QString& msg)
{
    setLog(msg);
}

void MainWindow::setLog(const QString log)
{
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formatedTime = currentDateTime.toString("[yyyy-MM-dd-hh:mm:ss]");
    ui->ted_log->append(formatedTime + ": " + log);
}

void MainWindow::saveDetectionResult(const cv::Mat& image, const QString& save_path)
{
    if (image.empty()) {
        return;
    }
    cv::imwrite(save_path.toStdString(), image);
    qDebug() << "Saved detection result:" << save_path;
}

void MainWindow::on_clear()
{
    if (isBatchRunning) {
        QMessageBox::warning(this, "Warning", "Please wait for batch test to complete!");
        return;
    }

    current_image.release();
    result_image.release();
    ui->lbl_image->clear();
    ui->lbl_image->setText("Please load an image");
    ui->ted_log->clear();
    setLog("Finished Clear!");
    ui->btn_infer->setEnabled(false);
    ui->btn_Batch->setEnabled(false);
}

void MainWindow::on_gpuCheck(int state)
{
    use_gpu = (state == Qt::Checked);
    std::cout << "GPU mode changed: " << (use_gpu ? "Enabled" : "Disabled") << std::endl;
    if (yoloWorker != nullptr && !model_path.isEmpty()) {
        int reply = QMessageBox::question(this, "Restart Model",
            "GPU/CPU mode changed. Restart model to apply changes?",
            QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            initializeModel(model_path);
        } else {
            ui->ckb_useGPU->setChecked(!use_gpu);
        }
    }
}

void MainWindow::displayImage(const cv::Mat& image)
{
    if (image.empty()) return;
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows,
                static_cast<int>(rgb.step),
                QImage::Format_RGB888);

    QPixmap pixmap = QPixmap::fromImage(qimg);
    ui->lbl_image->setPixmap(pixmap.scaled(
        ui->lbl_image->size(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    ));
}

bool MainWindow::loadConfigFile(const QString& model_path) {
    QFileInfo model_info(model_path);
    QString config_path = model_info.dir().filePath("infer_cfg.json");
    if (!QFile::exists(config_path)) {
        qDebug() << "Config file not found, using default size: 640x640";
        input_width = 640;
        input_height = 640;
        return false;
    }

    QFile config_file(config_path);
    if (!config_file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Failed to open config file: " << config_path;
        input_width = 640;
        input_height = 640;
        return false;
    }

    QByteArray config_data = config_file.readAll();
    config_file.close();

    QJsonParseError parse_error;
    QJsonDocument doc = QJsonDocument::fromJson(config_data, &parse_error);

    if (parse_error.error != QJsonParseError::NoError) {
        qDebug() << "JSON parse error: " << parse_error.errorString();
        input_width = 640;
        input_height = 640;
        return false;
    }

    QJsonObject json = doc.object();

    if (json.contains("target_size") && json["target_size"].isArray()) {
        QJsonArray size_array = json["target_size"].toArray();
        if (size_array.size() >= 2) {
            input_width = size_array[0].toInt();
            input_height = size_array[1].toInt();
            qDebug() << "Config loaded - Input size: " << input_width << "x" << input_height;
            return true;
        }
    }

    input_width = 640;
    input_height = 640;
    return false;
}
