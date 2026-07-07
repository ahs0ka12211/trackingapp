#include "Tracker.h"
#include <QDebug>

Tracker::Tracker(cv::VideoCapture& cap) : _cap(cap)
{
    _mog2         = cv::createBackgroundSubtractorMOG2(200, 50, false);
    _mog2tracking = cv::createBackgroundSubtractorMOG2(500, 60, false);
}

// ─────────────────────────────────────────────────────────────
//  Публичный интерфейс
// ─────────────────────────────────────────────────────────────

bool Tracker::findObject(const cv::Mat& frame)
{
    if (frame.empty()) return false;
    return _detecting ? detectPhase(frame) : trackPhase(frame);
}

void Tracker::reset()
{
    _mog2         = cv::createBackgroundSubtractorMOG2(200, 50, false);
    _mog2tracking = cv::createBackgroundSubtractorMOG2(500, 60, false);
    _hist.release();
    _prevGray.release();
    _detectionCount  = 0;
    _accumulatedMask = cv::Mat();
    _detecting       = true;
    _initialized     = false;
    _bbox            = cv::Rect();
    _smoothedBbox    = cv::Rect();
    _locked          = false;
    _start           = cv::Point();
    _end             = cv::Point();
}

// ─────────────────────────────────────────────────────────────
//  Общий метод получения маски движения
//  Используется и в detectPhase, и в trackPhase
// ─────────────────────────────────────────────────────────────

cv::Mat Tracker::getMotionMask(const cv::Mat& stabilized, bool forTracking)
{
    // Уменьшаем до HD если кадр больше
    cv::Mat input = stabilized;
    double scale = 1.0;
    if (stabilized.cols > 1280) {
        scale = 1280.0 / stabilized.cols;
        cv::resize(stabilized, input, cv::Size(), scale, scale);
    }

    cv::Mat mask;
    if (forTracking)
        _mog2tracking->apply(input, mask);
    else
        _mog2->apply(input, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Масштабируем маску обратно если уменьшали
    if (scale < 1.0)
        cv::resize(mask, mask, stabilized.size(), 0, 0, cv::INTER_NEAREST);

    return mask;
}

// ─────────────────────────────────────────────────────────────
//  Фаза 1 — детекция объекта через MOG2
// ─────────────────────────────────────────────────────────────

bool Tracker::detectPhase(const cv::Mat& frame)
{
    cv::Mat stabilized = stabilizeFrame(frame);
    cv::Mat mask = getMotionMask(stabilized, false);

    if (_accumulatedMask.empty())
        _accumulatedMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::bitwise_or(_accumulatedMask, mask, _accumulatedMask);

    _detectionCount++;
    if (_detectionCount < DETECTION_FRAMES)
        return false;

    cv::Rect detected = largestContourRect(_accumulatedMask);

    if (detected.empty()) {
        qDebug() << "[Tracker] объект не обнаружен, повторяем детекцию";
        _detectionCount  = 0;
        _accumulatedMask = cv::Mat();
        return false;
    }

    _bbox = detected;
    buildHistogram(frame, _bbox);

    _start       = _bbox.tl();
    _end         = _bbox.br();
    _detecting   = false;
    _initialized = true;

    qDebug() << "[Tracker] объект найден, bbox:"
             << detected.x << detected.y
             << detected.width << "x" << detected.height;

    return true;
}

// ─────────────────────────────────────────────────────────────
//  Фаза 2 — трекинг через CamShift + мониторинг новых объектов
// ─────────────────────────────────────────────────────────────

bool Tracker::trackPhase(const cv::Mat& frame)
{
    cv::Mat stabilized = stabilizeFrame(frame);
    cv::Mat motionMask = getMotionMask(stabilized, true);

    // Оцениваем общую активность фона (0.0 — 1.0)
    double totalMotion = (double)cv::countNonZero(motionMask)
                        / (motionMask.rows * motionMask.cols);

    // ── Шаг 1: CamShift ──────────────────────────────────────
    cv::Mat hsv;
    cv::cvtColor(stabilized, hsv, cv::COLOR_BGR2HSV);

    cv::Mat rangeMask;
    cv::inRange(hsv, cv::Scalar(0, 15, 15), cv::Scalar(180, 255, 255), rangeMask);

    cv::Mat backProj;
    int    channels[]     = {0, 1};
    float  hRange[]       = {0, 180};
    float  sRange[]       = {0, 256};
    const float* ranges[] = {hRange, sRange};
    cv::calcBackProject(&hsv, 1, channels, _hist, backProj, ranges);
    cv::bitwise_and(backProj, rangeMask, backProj);

    int dilateSize = (totalMotion > 0.15) ? 5 : 15;
    cv::Mat dilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(dilateSize, dilateSize));
    cv::Mat dilatedMotion;
    cv::dilate(motionMask, dilatedMotion, dilateKernel);
    cv::bitwise_and(backProj, dilatedMotion, backProj);

    cv::TermCriteria termCrit(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
        CAMSHIFT_ITERATIONS, CAMSHIFT_EPSILON
    );

    cv::RotatedRect rotRect = cv::CamShift(backProj, _bbox, termCrit);
    _bbox = rotRect.boundingRect();

    cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    _bbox &= frameRect;

    if (_bbox.area() < MIN_CONTOUR_AREA) {
        qDebug() << "[Tracker] объект потерян, возврат к детекции";
        reset();
        return false;
    }

    // ── Сглаживание ───────────────────────────────────────────
    const double alpha = 0.7;
    if (!_smoothedBbox.empty()) {
        _smoothedBbox.x      = alpha * _smoothedBbox.x      + (1-alpha) * _bbox.x;
        _smoothedBbox.y      = alpha * _smoothedBbox.y      + (1-alpha) * _bbox.y;
        _smoothedBbox.width  = alpha * _smoothedBbox.width  + (1-alpha) * _bbox.width;
        _smoothedBbox.height = alpha * _smoothedBbox.height + (1-alpha) * _bbox.height;
    } else {
        _smoothedBbox = _bbox;
    }

    // ── Шаг 2: поиск более активного объекта (только без фиксации) ──
    if (!_locked) {
        cv::Mat currentRegion   = motionMask(_bbox & frameRect);
        double  candidateArea   = 0;
        cv::Rect candidateRect  = largestContourRect(motionMask, &candidateArea);
        double candidateDensity = candidateArea / (candidateRect.area() + 1);
        double currentDensity   = (double)cv::countNonZero(currentRegion)
                                  / (_bbox.area() + 1);

        // Фон при быстром движении даёт высокую totalMotion —
        // в таком случае вообще не переключаемся
        bool backgroundTooActive = totalMotion > 0.4;

        bool isNewObject    = !candidateRect.empty();
        bool notOverlapping = (candidateRect & _bbox).area()
                              < candidateRect.area() * 0.3;
        bool isMoreActive   = candidateDensity > currentDensity * SWITCH_RATIO;
        bool isCompact      = candidateArea / (candidateRect.area() + 1) > 0.3;
        double sizeRatio    = (double)candidateRect.area() / (_bbox.area() + 1);
        bool isSimilarSize  = sizeRatio < 4.0 && sizeRatio > 0.25;
        bool currentIsStill = currentDensity < 0.05;

        if (!backgroundTooActive && isNewObject && notOverlapping
            && isMoreActive && isCompact && isSimilarSize && currentIsStill) {
            qDebug() << "[Tracker] переключаемся на новый объект";
            _bbox         = candidateRect;
            _smoothedBbox = candidateRect;
            buildHistogram(frame, _bbox);
        }
    }

    _start = _smoothedBbox.tl();
    _end   = _smoothedBbox.br();
    return true;
}

// ─────────────────────────────────────────────────────────────
//  Строим HSV-гистограмму объекта по bbox
// ─────────────────────────────────────────────────────────────

void Tracker::buildHistogram(const cv::Mat& frame, const cv::Rect& roi)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat roiHsv = hsv(roi);

    cv::Mat mask;
    cv::inRange(roiHsv, cv::Scalar(0, 15, 15), cv::Scalar(180, 255, 255), mask);

    // 2D гистограмма: H (64 бина) × S (64 бина)
    int    histSize[]  = {64, 64};
    float  hRange[]    = {0, 180};
    float  sRange[]    = {0, 256};
    const float* ranges[] = {hRange, sRange};
    int    channels[]  = {0, 1}; // H и S каналы

    cv::calcHist(&roiHsv, 1, channels, mask, _hist, 2, histSize, ranges);
    cv::normalize(_hist, _hist, 0, 255, cv::NORM_MINMAX);
}

// ─────────────────────────────────────────────────────────────
//  Стабилизация кадра — компенсация движения камеры
// ─────────────────────────────────────────────────────────────

cv::Mat Tracker::stabilizeFrame(const cv::Mat& frame)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

     if (frame.cols > 1280) {
        double scale = 1280.0 / frame.cols;
        cv::resize(gray, gray, cv::Size(), scale, scale);
    }
    

    if (_prevGray.empty()) {
        _prevGray = gray.clone();
        return frame;
    }

    std::vector<cv::Point2f> prevPts, currPts;
    cv::goodFeaturesToTrack(_prevGray, prevPts, 100, 0.01, 30);

    if (prevPts.empty()) {
        _prevGray = gray.clone();
        return frame;
    }

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(_prevGray, gray, prevPts, currPts, status, err);

    std::vector<cv::Point2f> goodPrev, goodCurr;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            goodPrev.push_back(prevPts[i]);
            goodCurr.push_back(currPts[i]);
        }
    }

    _prevGray = gray.clone();

    if (goodPrev.size() < 4)
        return frame;

    cv::Mat transform = cv::estimateAffinePartial2D(
        goodPrev, goodCurr, cv::noArray(), cv::RANSAC
    );

    if (transform.empty())
        return frame;

    transform.at<double>(0, 2) = -transform.at<double>(0, 2);
    transform.at<double>(1, 2) = -transform.at<double>(1, 2);

    cv::Mat stabilized;
    cv::warpAffine(frame, stabilized, transform, frame.size());
    return stabilized;
}

// ─────────────────────────────────────────────────────────────
//  Самый крупный контур на маске
// ─────────────────────────────────────────────────────────────

cv::Rect Tracker::largestContourRect(const cv::Mat& mask, double* outArea) const
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Площадь всего кадра
    double frameArea = mask.rows * mask.cols;

    double   maxArea = 0;
    cv::Rect best;

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area > frameArea * 0.4) continue;
        if (area < MIN_CONTOUR_AREA)  continue;

        cv::Rect r = cv::boundingRect(c);

        // Фильтр по соотношению сторон: не более 5:1
        double aspect = (double)std::max(r.width, r.height)
                    / (std::min(r.width, r.height) + 1);
        if (aspect > 5.0) continue;

        if (area > maxArea) {
            maxArea = area;
            best    = r;
        }
    }

    if (outArea) *outArea = maxArea;
    return best;
}
