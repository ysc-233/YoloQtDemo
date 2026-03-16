# YOLO Object Detection Demo (Qt + ONNX Runtime)

本项目是一个基于 C++、Qt 和 ONNX Runtime 的跨平台目标检测演示程序。它支持加载自定义的 YOLO ONNX 模型，利用 GPU (CUDA) 或 CPU 进行推理，并提供单张图像检测和批量图像测试功能。

## 项目结构

```text
yolo_qt_demo/
├── models/                     # 模型存放目录
│   └── test.onnx               # 默认测试模型
├── main.cpp                    # 程序入口，注册元类型
├── mainwindow.h / .cpp         # 主窗口逻辑 (UI 交互、线程管理)
├── mainwindow.ui               # Qt Designer 界面文件
├── yoloinference.h / .cpp      # 核心推理类 (ONNX Session 管理、预处理、后处理)
└── CMakeLists.txt              # CMake 构建配置
```

## 主要功能

- **多后端支持**: 自动检测并优先使用 CUDA (GPU) 加速，若不可用则自动回退到 CPU。
- **动态模型加载**: 支持运行时选择 .onnx 模型文件。
- **智能配置读取**: 自动读取模型同目录下的 infer_cfg.json 获取输入尺寸 (target_size)，若无配置则默认 640x640。
- **图像处理**:
  - 自动 Letterbox 缩放与填充 (保持宽高比)。
  - 实时绘制检测框、类别名称及置信度。
  - 结果自动保存至 detRes 文件夹。
- **多线程架构**:
  - 使用 QThread 将耗时的推理任务移至工作线程，确保 UI 界面流畅不卡顿。
  - 通过信号槽机制 (signals/slots) 进行线程间通信。
- **批量测试**: 支持选择文件夹进行批量图片推理，实时显示进度条、成功/失败统计及平均 FPS。
- **日志系统**: 实时显示模型加载、推理状态、错误信息及详细检测结果。

## 技术栈

- **语言**: C++17 (或更高)
- **GUI 框架**: Qt 5 / Qt 6
- **推理引擎**: Microsoft ONNX Runtime (onnxruntime_cxx_api)
- **图像处理**: OpenCV (opencv2/opencv.hpp)
- **加速后端**: NVIDIA CUDA (通过 ONNX Runtime CUDA Execution Provider)
- **构建工具**: CMake

## 核心模块说明

### 1. YOLOInference (推理核心)
位于 yoloinference.cpp/h。
- **初始化**: 创建 Ort::Env 和 Ort::Session。若启用 GPU，会配置 OrtCUDAProviderOptions。代码中包含对 CUDA 设备可用性的检查，若失败会自动降级为 CPU。
- **预处理 (prepareInput)**:
  - 执行 Letterbox 变换 (保持比例缩放并填充灰色边框)。
  - BGR 转 RGB。
  - 归一化 (0-255 -> 0.0-1.0)。
  - HWC 转 CHW 格式以适配 ONNX 输入。
- **推理 (runInference)**: 调用 session->Run() 执行计算，并使用 chrono 记录耗时。
- **后处理 (postprocess)**:
  - 解码输出张量 (坐标反映射、置信度筛选)。
  - 执行 NMS (非极大值抑制) 去除重叠框。
- **注意**: 由于 CUDA 上下文初始化和 JIT 编译，首次推理（或长时间空闲后的第一次）耗时可能较长。当前在加载完模型后使用黑色图像进行预热。

### 2. MainWindow (用户界面)
位于 mainwindow.cpp/h。
- **线程管理**: 负责创建 QThread，实例化 YOLOInference 对象并移动至该线程 (moveToThread)。
- **信号连接**: 连接推理完成的信号以更新 UI 图片、日志和统计信息。所有跨线程信号均使用 Qt::QueuedConnection。
- **配置加载**: loadConfigFile 函数解析 JSON 配置文件以确定模型输入分辨率。
- **异常处理**: 当 GPU 初始化失败时，自动尝试切换至 CPU 模式并提示用户。
- **批量处理**: 管理批量测试的状态标志 (isBatchRunning)，防止重复启动，并汇总最终统计信息。

## 配置文件格式 (infer_cfg.json)

模型文件夹中可选的配置文件，用于指定输入尺寸：

```json
{
  "target_size": [640, 640]
}
```

如果不存在此文件，程序将默认使用 640x640。

## 编译与运行

### 前置依赖
1. **Qt**: 安装 Qt Creator 及 Qt 库 (Widgets 模块)。
2. **OpenCV**: 编译安装或通过包管理器安装 (需包含 opencv_core, opencv_imgproc, opencv_dnn 等)。
3. **ONNX Runtime**:
   - 下载 ONNX Runtime C++ API (含 GPU 支持版本如需 CUDA)。
   - 确保 onnxruntime_cxx_api.h 和动态库 (.dll/.so/.dylib) 在编译路径中。
4. **CUDA Toolkit** (可选): 如需 GPU 加速，需安装匹配的 NVIDIA 驱动和 CUDA Toolkit（推荐CUDA11.8）。

### CMake 构建步骤

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt -DONNXRUNTIME_DIR=/path/to/onnxruntime
make
./yolo_qt_demo
```

*(具体 CMake 参数请参考项目中的 CMakeLists.txt)*

## 使用指南

1. **启动程序**: 运行编译后的可执行文件。
2. **加载模型**: 点击 "Load Model" 按钮，选择 .onnx 文件 (建议选择 models/ 目录下的模型)。
   - 程序会自动尝试启用 GPU，若失败则提示并切换至 CPU。
3. **加载图片**: 点击 "Load Image" 选择单张图片，或 "Set Image Dir" 选择文件夹。
4. **开始推理**:
   - 单张：点击 "Infer"。
   - 批量：点击 "Batch Test"。
5. **查看结果**: 检测结果将直接显示在主窗口，并在程序目录下的 detRes 文件夹中保存带框图片。日志窗口会显示耗时和 FPS。

## 注意事项

- **首次推理延迟**: 由于 CUDA 上下文初始化和 JIT 编译，第一次推理（或长时间空闲后的第一次）耗时可能较长，属正常现象。
- **显存占用**: GPU 模式下会预分配显存池，请确保显存充足。
- **线程安全**: OpenCV 的 cv::Mat 已通过 qRegisterMetaType 注册，支持在 Qt 信号槽中跨线程传递。
- **模型切换**: 更改 GPU/CPU 选项后，需要重新加载模型才能生效。