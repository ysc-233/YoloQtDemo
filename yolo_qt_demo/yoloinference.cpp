#include "yoloinference.h"
#include <QDateTime>
#include <chrono>
#include <fstream>
#include <QFile>
#include <QDir>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void YOLOInference::emitLog(const std::string& msg, bool isError) {
    QString qMsg = QString::fromStdString(msg);
    if (isError) {
        qMsg = "[ERROR] " + qMsg;
    }
    emit logSignal(qMsg);
}

cv::Mat letterbox(const cv::Mat& image, cv::Size new_shape) {
    cv::Mat resized;
    float ratio = std::min(new_shape.width / (float)image.cols, new_shape.height / (float)image.rows);
    int new_unpad_w = int(image.cols * ratio);
    int new_unpad_h = int(image.rows * ratio);
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    int dw = new_shape.width - new_unpad_w;
    int dh = new_shape.height - new_unpad_h;

    int top = int(dh / 2);
    int bottom = dh - top;
    int left = int(dw / 2);
    int right = dw - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return padded;
}

YOLOInference::YOLOInference(const std::string& model_path, bool gpu,
                             float conf_thresh, float nms_thresh,
                             int input_w, int input_h,
                             QObject *parent)
    : QObject(parent),
      use_gpu(gpu), conf_threshold(conf_thresh), nms_threshold(nms_thresh),
      input_width(input_w), input_height(input_h) {

    emitLog("Loading model: " + model_path);
    emitLog("Using GPU: " + std::string(use_gpu ? "Yes" : "No"));
    emitLog("Input size: " + std::to_string(input_width) + "x" + std::to_string(input_height));

    std::ifstream file(model_path);
    if (!file.good()) {
        emitLog("ERROR: Model file not found: " + model_path, true);
        return;
    }
    file.close();

    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOInference");
        session_options = std::make_unique<Ort::SessionOptions>();
        session_options->SetIntraOpNumThreads(1);
        session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
#ifdef USE_CUDA
            cudaError_t cuda_status = cudaSetDevice(0);
            if (cuda_status != cudaSuccess) {
                emitLog("WARNING: CUDA device not available, falling back to CPU", true);
                use_gpu = false;
            } else {
                std::vector<std::string> providers = Ort::GetAvailableProviders();
                bool cuda_available = false;
                for (const auto & provider : providers) {
                    if (std::string(provider) == "CUDAExecutionProvider") {
                        cuda_available = true;
                        break;
                    }
                }
                if (cuda_available) {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = 0;
                    session_options->AppendExecutionProvider_CUDA(cuda_options);
                    emitLog("CUDA Execution Provider enabled");
                } else {
                    emitLog("WARNING: CUDA Execution Provider not available, falling back to CPU", true);
                    use_gpu = false;
                }
            }
#else
            emitLog("WARNING: CUDA support not compiled, falling back to CPU", true);
            use_gpu = false;
#endif
        }

        if (!use_gpu) {
            emitLog("CPU Execution Provider enabled (default)");
        }

        std::wstring wide_model_path(model_path.begin(), model_path.end());
        session = std::make_unique<Ort::Session>(*env, wide_model_path.c_str(), *session_options);

        if (session == nullptr) {
            emitLog("ERROR: Failed to create ONNX session!", true);
            return;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(std::string(input_name.get()));
            emitLog("Input " + std::to_string(i) + ": " + input_names.back());
        }

        size_t num_output_nodes = session->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(std::string(output_name.get()));
            emitLog("Output " + std::to_string(i) + ": " + output_names.back());
        }

        emitLog("Model loaded successfully!");
        emitLog("Running on: " + std::string(use_gpu ? "GPU" : "CPU"));
        model_loaded = true;
    } catch (const Ort::Exception & e) {
        emitLog("ERROR: ONNX Runtime exception: " + std::string(e.what()), true);
        session = nullptr;
        return;
    }
}

YOLOInference::~YOLOInference() {
    emitLog("YOLOInference destructor called");
}

void YOLOInference::prepareInput(const cv::Mat& image, std::vector<float>& input_tensor, cv::Mat& padded_img) {
    if (image.empty()) {
        emitLog("ERROR: Input image is empty!", true);
        return;
    }
    padded_img = letterbox(image, cv::Size(input_width, input_height));

    cv::Mat rgb;
    cv::cvtColor(padded_img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    input_tensor.resize(3 * input_height * input_width);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < channels[c].rows; h++) {
            for (int w = 0; w < channels[c].cols; w++) {
                input_tensor[c * input_height * input_width + h * input_width + w] =
                        channels[c].at<float>(h, w);
            }
        }
    }
}

std::vector<Detection> YOLOInference::runInference(const cv::Mat& image, double& elapsed_ms) {
    std::vector<Detection> detections;
    elapsed_ms = 0.0;

    if (!model_loaded || session == nullptr) {
        emitLog("ERROR: Model not loaded!", true);
        return detections;
    }
    if (image.empty()) {
        emitLog("ERROR: Input image is empty!", true);
        return detections;
    }

    auto start = std::chrono::high_resolution_clock::now();

    try {
        std::vector<float> input_tensor;
        cv::Mat padded_img;
        prepareInput(image, input_tensor, padded_img);

        if (input_tensor.empty()) {
            emitLog("ERROR: Input tensor is empty after prepareInput!", true);
            return detections;
        }

        std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_tensor.data(),
                    input_tensor.size(),
                    input_shape.data(),
                    input_shape.size());

        std::vector<const char*> input_name_ptrs;
        for (const auto & name : input_names) input_name_ptrs.push_back(name.c_str());

        std::vector<const char*> output_name_ptrs;
        for (const auto & name : output_names) output_name_ptrs.push_back(name.c_str());

        auto output_tensors = session->Run(
                    Ort::RunOptions{nullptr},
                    input_name_ptrs.data(),
                    &input_tensor_ort,
                    1,
                    output_name_ptrs.data(),
                    output_name_ptrs.size());

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_size = type_info.GetElementCount();
        auto shape = type_info.GetShape();

        if (shape.size() != 3) {
            emitLog("ERROR: Unexpected output shape size!", true);
            return detections;
        }

        int features = shape[1];
        int num_anchors = shape[2];

        std::vector<float> output_vector(output_data, output_data + output_size);
        detections = postprocess(image, output_vector, features, num_anchors, padded_img);

    } catch (const Ort::Exception & e) {
        emitLog("ERROR: Inference failed - " + std::string(e.what()), true);
    }

    auto end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return detections;
}

cv::Mat YOLOInference::drawDetectionsOnImage(const cv::Mat& image, const std::vector<Detection>& detections) {
    cv::Mat result = image.clone();
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255)
    };

    for (const auto & det : detections) {
        cv::Scalar color = colors[det.class_id % colors.size()];
        cv::rectangle(result, det.bbox, color, 2);

        std::string label = det.class_name + ": " +
                std::to_string(static_cast<int>(det.confidence * 100)) + "% (" +
                std::to_string(static_cast<int>(det.bbox.x)) + ", " +
                std::to_string(static_cast<int>(det.bbox.y)) + ")";
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.6, 2, &baseline);
        cv::Point text_origin(det.bbox.x, det.bbox.y - 10);
        cv::rectangle(result,
                     cv::Point(text_origin.x, text_origin.y - text_size.height),
                      cv::Point(text_origin.x + text_size.width, text_origin.y + baseline),
                     color, -1);

        cv::putText(result, label, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 2);
    }

    return result;
}

void YOLOInference::processImage(const cv::Mat& image) {
    double elapsed_ms = 0.0;
    std::vector<Detection> detections = runInference(image, elapsed_ms);

    cv::Mat result_image = drawDetectionsOnImage(image, detections);

    emit inferenceFinished(detections, elapsed_ms, result_image);
}

void YOLOInference::processBatch(const QStringList& image_paths, const QString& result_dir) {
    if (image_paths.isEmpty()) {
        emit batchFinished(0, 0, 0, 0.0);
        return;
    }

    QDir().mkpath(result_dir);

    int total = image_paths.size();
    int success = 0;
    int failed = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < image_paths.size(); i++) {
        QString file_path = image_paths[i];
        cv::Mat img = cv::imread(file_path.toStdString());

        if (img.empty()) {
            failed++;
            cv::Mat result_image;
            emit batchProgress(i + 1, total, QString("Failed: %1").arg(file_path),result_image);
            continue;
        }

        double elapsed_ms = 0.0;
        std::vector<Detection> detections = runInference(img, elapsed_ms);

        cv::Mat result = drawDetectionsOnImage(img, detections);

        QFileInfo fi(file_path);
        QString base_name = fi.completeBaseName();
        QString save_path = result_dir + "/" + base_name + "_det.jpg";

        cv::imwrite(save_path.toStdString(), result);

        success++;
        cv::Mat result_image = drawDetectionsOnImage(img, detections);
        emit batchProgress(i + 1, total, QString("Processing: %1/%2").arg(i + 1).arg(total),result_image);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    double total_ms = total_duration.count();
    float avg_fps = total > 0 && total_ms > 0 ? (total * 1000.0f / total_ms) : 0.0f;

    emit batchFinished(total, success, failed, avg_fps);
}

std::vector<Detection> YOLOInference::postprocess(const cv::Mat& image, const std::vector<float>& output,
                                                  int features, int num_anchors, const cv::Mat& padded_img) {
    std::vector<Detection> detections;
    int num_classes = features - 4;
    if (num_classes <= 0 || output.size() != static_cast<size_t>(features * num_anchors)) {
        emitLog("ERROR: Invalid output shape!", true);
        return detections;
    }
    float gain = std::min((float)input_width / image.cols, (float)input_height / image.rows);
    int pad_x = int((input_width - image.cols * gain) / 2);
    int pad_y = int((input_height - image.rows * gain) / 2);

    for (int a = 0; a < num_anchors; a++) {
        float cx = output[0 * features * num_anchors + 0 * num_anchors + a];
        float cy = output[0 * features * num_anchors + 1 * num_anchors + a];
        float w   = output[0 * features * num_anchors + 2 * num_anchors + a];
        float h = output[0 * features * num_anchors + 3 * num_anchors + a];

        int class_id = 0;
        float max_score = output[0 * features * num_anchors + 4 * num_anchors + a];
        for (int c = 1; c < num_classes; c++) {
            float score = output[0 * features * num_anchors + (4 + c) * num_anchors + a];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= conf_threshold) {
            float x1 = (cx - 0.5f * w);
            float y1 = (cy - 0.5f * h);
            float x2 = (cx + 0.5f * w);
            float y2 = (cy + 0.5f * h);

            x1 = (x1 - pad_x) / gain;
            y1 = (y1 - pad_y) / gain;
            x2 = (x2 - pad_x) / gain;
            y2 = (y2 - pad_y) / gain;

            x1 = std::max(0.0f, std::min(x1, (float)image.cols));
            y1 = std::max(0.0f, std::min(y1, (float)image.rows));
            x2 = std::max(0.0f, std::min(x2, (float)image.cols));
            y2 = std::max(0.0f, std::min(y2, (float)image.rows));

            Detection det;
            det.bbox = cv::Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1));
            det.confidence = max_score;
            det.class_id = class_id;

            if (class_id < static_cast<int>(classes.size())) {
                det.class_name = classes[class_id];
            } else {
                det.class_name = "class_" + std::to_string(class_id);
            }
            detections.push_back(det);
        }
    }

    return applyNMS(detections);
}

std::vector<Detection> YOLOInference::applyNMS(std::vector<Detection>& detections) {
    std::vector<Detection> final_detections;
    if (detections.empty()) {
        return final_detections;
    }
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }

    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        final_detections.push_back(detections[idx]);
    }
    return final_detections;
}
