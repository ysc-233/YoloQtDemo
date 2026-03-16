cmake_minimum_required(VERSION 3.10)
project(yolo_qt_demo VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# 1. Qt 配置
find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets Core OpenGL Gui REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets Core OpenGL Gui REQUIRED)

# 2. CUDA 选项（必须在 add_subdirectory 之前定义）
option(USE_CUDA "Enable CUDA GPU support" ON)  # 默认 OFF，如需 GPU 请手动开启
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA support disabled (CPU only)")
endif()

# 3. 第三方库（包含 OpenCV 和 ONNX Runtime 的配置）
add_subdirectory(ThirdParties)

# 4. OpenCV 配置
set(OPENCV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties/OpenCV)  # 假设 OpenCV 也在 ThirdParties 下

if(EXISTS ${OPENCV_ROOT})
    set(OpenCV_INCLUDE_DIRS
        ${OPENCV_ROOT}/include
        ${OPENCV_ROOT}/include/opencv2
    )

    # 获取所有 .lib 文件
    file(GLOB ALL_OPENCV_LIBS ${OPENCV_ROOT}/lib/*.lib)

    # 分离 Debug 和 Release 库
    set(OpenCV_LIBS_DEBUG "")
    set(OpenCV_LIBS_RELEASE "")
    foreach(LIB_PATH ${ALL_OPENCV_LIBS})
        get_filename_component(LIB_NAME ${LIB_PATH} NAME)
        if(LIB_NAME MATCHES "d\\.lib$")
            list(APPEND OpenCV_LIBS_DEBUG ${LIB_PATH})
        else()
            list(APPEND OpenCV_LIBS_RELEASE ${LIB_PATH})
        endif()
    endforeach()

    if(NOT OpenCV_LIBS_RELEASE)
        message(FATAL_ERROR "OpenCV Release libraries not found in ${OPENCV_ROOT}/lib")
    endif()
    if(NOT OpenCV_LIBS_DEBUG)
        message(WARNING "OpenCV Debug libraries not found. Using Release libraries for Debug configuration.")
        set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS_RELEASE})
    endif()

    message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV Release libs: ${OpenCV_LIBS_RELEASE}")
    message(STATUS "OpenCV Debug libs: ${OpenCV_LIBS_DEBUG}")
else()
    message(FATAL_ERROR "OpenCV not found: ${OPENCV_ROOT}")
endif()

# 5. 项目源文件
file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.ui
)
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*/ThirdParties/.*")
if(NOT PROJECT_SOURCES)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        "#include <QApplication>\n"
        "#include <QMainWindow>\n"
        "int main(int argc, char *argv[]){\n"
        "QApplication app(argc, argv);\n"
        "QMainWindow w;\n"
        "w.show();\n"
        "return app.exec();}\n"
    )
    set(PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 6. 可执行程序
add_executable(yolo_qt_demo WIN32 ${PROJECT_SOURCES})

# 7. 链接库（使用生成器表达式代替 debug/optimized 关键字）
target_link_libraries(yolo_qt_demo PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::OpenGL
    Qt${QT_VERSION_MAJOR}::Gui

    # OpenCV 链接（每个库单独用生成器表达式）
    $<$<CONFIG:Debug>:${OpenCV_LIBS_DEBUG}>
    $<$<CONFIG:Release>:${OpenCV_LIBS_RELEASE}>

    # ONNX Runtime 链接（由 ThirdParties 导出，已使用生成器表达式）
    ${ONNXRUNTIME_LIBRARY}
)

if(USE_CUDA)
    target_link_libraries(yolo_qt_demo PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
endif()

# 8. 包含目录
target_include_directories(yolo_qt_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# 9. 编译定义
target_compile_definitions(yolo_qt_demo PRIVATE
    QT_DEPRECATED_WARNINGS
)

# 10. 输出目录
set_target_properties(yolo_qt_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# 11. 复制 ONNX Runtime 所有 DLL 到输出目录
if(DEFINED ONNXRUNTIME_LIB_DIR AND EXISTS "${ONNXRUNTIME_LIB_DIR}")
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_LIB_DIR}/*.dll")
    foreach(DLL ${ONNXRUNTIME_DLLS})
        add_custom_command(TARGET yolo_qt_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}"
            $<TARGET_FILE_DIR:yolo_qt_demo>
        )
    endforeach()
    message(STATUS "Copied ONNX Runtime DLLs from: ${ONNXRUNTIME_LIB_DIR}")
else()
    message(WARNING "ONNXRUNTIME_LIB_DIR not defined or directory not found, cannot copy ONNX Runtime DLLs.")
endif()

# 12. 复制 OpenCV DLL
set(OPENCV_BIN_DIR "${OPENCV_ROOT}/lib")
if(EXISTS ${OPENCV_BIN_DIR})
    file(GLOB OPENCV_DLLS ${OPENCV_BIN_DIR}/*.dll)
    foreach(DLL ${OPENCV_DLLS})
        add_custom_command(TARGET yolo_qt_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}"
            $<TARGET_FILE_DIR:yolo_qt_demo>
        )
    endforeach()
endif()

# 13. Qt5 不需要 qt_finalize_executable，但保留 Qt6 兼容
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(yolo_qt_demo)
endif()
