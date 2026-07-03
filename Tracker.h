#pragma once
#include <opencv2/opencv.hpp>

class Tracker
{
public:
    static constexpr int    DETECTION_FRAMES   = 10;
    static constexpr double MIN_CONTOUR_AREA   = 500.0;
    static constexpr int    CAMSHIFT_ITERATIONS = 10;
    static constexpr double CAMSHIFT_EPSILON   = 1.0;

    // Во сколько раз новый объект должен быть "активнее" текущего,
    // чтобы трекер переключился на него
    static constexpr double SWITCH_RATIO = 2.0;

    explicit Tracker(cv::VideoCapture& cap);
    ~Tracker() = default;

    bool findObject(const cv::Mat& frame);
    void reset();

    bool isInitialized() const { return _initialized; }
    bool isDetecting()   const { return _detecting; }

    cv::Point get_start() const { return _start; }
    cv::Point get_end()   const { return _end; }

private:
    bool detectPhase(const cv::Mat& frame);
    bool trackPhase(const cv::Mat& frame);

    // Возвращает самый крупный контур на маске и его площадь
    cv::Rect largestContourRect(const cv::Mat& mask,
                                double* outArea = nullptr) const;

    void    buildHistogram(const cv::Mat& frame, const cv::Rect& roi);
    cv::Mat stabilizeFrame(const cv::Mat& frame);

    // Получаем маску движения для текущего кадра (стабилизированного)
    cv::Mat getMotionMask(const cv::Mat& stabilized, bool forTracking);

    cv::VideoCapture& _cap;

    // MOG2 используется и в детекции, и в трекинге (параллельно)
    cv::Ptr<cv::BackgroundSubtractorMOG2> _mog2;        // для детекции
    cv::Ptr<cv::BackgroundSubtractorMOG2> _mog2tracking; // для мониторинга во время трекинга

    cv::Mat _prevGray;
    cv::Mat _hist;

    int     _detectionCount  = 0;
    cv::Mat _accumulatedMask;

    bool _detecting   = true;
    bool _initialized = false;

    cv::Rect  _bbox;
    cv::Point _start;
    cv::Point _end;
};
