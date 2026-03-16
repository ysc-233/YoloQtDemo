#ifndef YOLOINFERENCE_H
#define YOLOINFERENCE_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <QObject>
#include <vector>
#include <string>
#include <memory>
#include <QString>

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

class YOLOInference : public QObject {
    Q_OBJECT
public:
    explicit YOLOInference(const std::string& model_path, bool use_gpu = false,
                           float conf_threshold = 0.25f, float nms_threshold = 0.45f,
                           int input_width = 640, int input_height = 640,
                           QObject *parent = nullptr);
    ~YOLOInference();

    bool isLoaded() const { return model_loaded; }
    bool isUsingGPU() const { return use_gpu; }
    int getInputWidth() const { return input_width; }
    int getInputHeight() const { return input_height; }

public slots:

    void processImage(const cv::Mat& image);

    void processBatch(const QStringList& image_paths, const QString& result_dir);

signals:
    void logSignal(const QString& message);

    void inferenceFinished(const std::vector<Detection>& detections, double elapsed_ms, const cv::Mat& result_image);

    void batchProgress(int current, int total, const QString& status, const cv::Mat& result_image);

    void batchFinished(int total, int success, int failed, double avg_fps);

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    bool use_gpu;
    float conf_threshold;
    float nms_threshold;
    int input_width;
    int input_height;
    bool model_loaded = false;
    std::vector<std::string> classes = { "bumps" };
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    void prepareInput(const cv::Mat& image, std::vector<float>& input_tensor, cv::Mat& padded_img);
    std::vector<Detection> postprocess(const cv::Mat& image, const std::vector<float>& output,
                                       int features, int num_anchors, const cv::Mat& padded_img);
    std::vector<Detection> applyNMS(std::vector<Detection>& detections);
    void emitLog(const std::string& msg, bool isError = false);

    std::vector<Detection> runInference(const cv::Mat& image, double& elapsed_ms);

    cv::Mat drawDetectionsOnImage(const cv::Mat& image, const std::vector<Detection>& detections);
};

#endif // YOLOINFERENCE_H
