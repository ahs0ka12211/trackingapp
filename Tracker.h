#pragma once
#include <opencv2/opencv.hpp>

class Tracker
{
public:
    static constexpr int    DETECTION_FRAMES    = 10;
    static constexpr double MIN_CONTOUR_AREA    = 300.0;
    static constexpr int    CAMSHIFT_ITERATIONS = 10;
    static constexpr double CAMSHIFT_EPSILON    = 1.0;
    static constexpr double SWITCH_RATIO        = 2.0;

    explicit Tracker(cv::VideoCapture& cap);
    ~Tracker() = default;

    bool findObject(const cv::Mat& frame);
    void reset();

    bool isInitialized() const { return _initialized; }
    bool isDetecting()   const { return _detecting; }
    bool isLocked()      const { return _locked; }

    // Включить/выключить фиксацию на текущем объекте
    void setLocked(bool locked) { _locked = locked; }

    cv::Point get_start() const { return _start; }
    cv::Point get_end()   const { return _end; }

private:
    bool detectPhase(const cv::Mat& frame);
    bool trackPhase(const cv::Mat& frame);

    cv::Rect largestContourRect(const cv::Mat& mask,
                                double* outArea = nullptr) const;

    void    buildHistogram(const cv::Mat& frame, const cv::Rect& roi);
    cv::Mat stabilizeFrame(const cv::Mat& frame);
    cv::Mat getMotionMask(const cv::Mat& stabilized, bool forTracking);

    cv::VideoCapture& _cap;

    cv::Ptr<cv::BackgroundSubtractorMOG2> _mog2;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _mog2tracking;

    cv::Mat _prevGray;
    cv::Mat _hist;

    int     _detectionCount  = 0;
    cv::Mat _accumulatedMask;

    bool _detecting   = true;
    bool _initialized = false;
    bool _locked      = false; // true = не переключаться на другие объекты

    cv::Rect  _bbox;
    cv::Rect  _smoothedBbox;
    cv::Point _start;
    cv::Point _end;
};
