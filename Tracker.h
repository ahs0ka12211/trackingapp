#pragma once
#include <opencv2/opencv.hpp>
// Только базовый OpenCV — opencv_contrib не нужен

class Tracker
{
public:
    static constexpr int    DETECTION_FRAMES = 10;
    static constexpr double MIN_CONTOUR_AREA = 500.0;

    // Параметры CamShift
    static constexpr int CAMSHIFT_ITERATIONS = 10;
    static constexpr double CAMSHIFT_EPSILON = 1.0;

    explicit Tracker(cv::VideoCapture& cap);
    ~Tracker() = default;

    // Вызывается на каждом кадре. true = объект найден/отслеживается.
    bool findObject(const cv::Mat& frame);

    // Сброс (новое видео или перемотка)
    void reset();

    bool isInitialized() const { return _initialized; }
    bool isDetecting()   const { return _detecting; }

    cv::Point get_start() const { return _start; }
    cv::Point get_end()   const { return _end; }

private:
    bool detectPhase(const cv::Mat& frame);
    bool trackPhase(const cv::Mat& frame);

    cv::Rect largestContourRect(const cv::Mat& mask) const;

    // Строим HSV-гистограмму объекта из начального bbox
    void buildHistogram(const cv::Mat& frame, const cv::Rect& roi);

    cv::VideoCapture& _cap;

    // MOG2 для фазы детекции
    cv::Ptr<cv::BackgroundSubtractorMOG2> _mog2;

    // Гистограмма HSV объекта (для CamShift)
    cv::Mat _hist;

    int     _detectionCount  = 0;
    cv::Mat _accumulatedMask;

    bool _detecting   = true;
    bool _initialized = false;

    cv::Rect  _bbox;
    cv::Point _start;
    cv::Point _end;
};
