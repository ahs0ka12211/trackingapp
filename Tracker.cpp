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
    _start           = cv::Point();
    _end             = cv::Point();
}

// ─────────────────────────────────────────────────────────────
//  Общий метод получения маски движения
//  Используется и в detectPhase, и в trackPhase
// ─────────────────────────────────────────────────────────────

cv::Mat Tracker::getMotionMask(const cv::Mat& stabilized, bool forTracking)
{
    cv::Mat mask;
    if (forTracking)
        _mog2tracking->apply(stabilized, mask);
    else
        _mog2->apply(stabilized, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

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
    // ── Шаг 1: CamShift обновляет текущий bbox ───────────────
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat rangeMask;
    cv::inRange(hsv, cv::Scalar(0, 30, 30), cv::Scalar(180, 255, 255), rangeMask);

    cv::Mat backProj;
    int         channels[] = {0};
    float       hRange[]   = {0, 180};
    const float* ranges[]  = {hRange};
    cv::calcBackProject(&hsv, 1, channels, _hist, backProj, ranges);
    cv::bitwise_and(backProj, rangeMask, backProj);

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

    // ── Шаг 2: параллельно гоним MOG2, ищем более активный объект ──
    cv::Mat stabilized = stabilizeFrame(frame);
    cv::Mat motionMask = getMotionMask(stabilized, true);

    // Считаем площадь движения внутри текущего bbox
    cv::Mat currentRegion = motionMask(_bbox & frameRect);
    double currentMotion = (double)cv::countNonZero(currentRegion) / (_bbox.area() + 1);

    // Ищем самый крупный движущийся объект во всём кадре
    double    candidateArea    = 0;
    cv::Rect  candidateRect    = largestContourRect(motionMask, &candidateArea);
    double    candidateDensity = candidateArea / (candidateRect.area() + 1);
    double    currentDensity   = (double)cv::countNonZero(currentRegion) / (_bbox.area() + 1);

    // Проверяем: кандидат существует, не совпадает с текущим объектом
    // и активнее текущего в SWITCH_RATIO раз
    bool isNewObject   = !candidateRect.empty();
    bool notOverlapping = (candidateRect & _bbox).area() < candidateRect.area() * 0.3;
    bool isMoreActive = candidateDensity > currentDensity * SWITCH_RATIO;

    if (isNewObject && notOverlapping && isMoreActive) {
        qDebug() << "[Tracker] обнаружен более активный объект, переключаемся"
                 << "| старый:" << currentMotion
                 << "| новый:"  << candidateArea;

        // Переключаемся: перестраиваем гистограмму под новый объект
        _bbox = candidateRect;
        buildHistogram(frame, _bbox);
    }

    _start = _bbox.tl();
    _end   = _bbox.br();
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
    cv::inRange(roiHsv, cv::Scalar(0, 30, 30), cv::Scalar(180, 255, 255), mask);

    int         histSize   = 64;
    float       hRange[]   = {0, 180};
    const float* ranges[]  = {hRange};
    int         channels[] = {0};

    cv::calcHist(&roiHsv, 1, channels, mask, _hist, 1, &histSize, ranges);
    cv::normalize(_hist, _hist, 0, 255, cv::NORM_MINMAX);
}

// ─────────────────────────────────────────────────────────────
//  Стабилизация кадра — компенсация движения камеры
// ─────────────────────────────────────────────────────────────

cv::Mat Tracker::stabilizeFrame(const cv::Mat& frame)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    if (_prevGray.empty()) {
        _prevGray = gray.clone();
        return frame;
    }

    std::vector<cv::Point2f> prevPts, currPts;
    cv::goodFeaturesToTrack(_prevGray, prevPts, 200, 0.01, 30);

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

        // Игнорируем контуры больше 40% кадра — это шум/фон, не объект
        if (area > frameArea * 0.4) continue;

        if (area > maxArea && area >= MIN_CONTOUR_AREA) {
            maxArea = area;
            best    = cv::boundingRect(c);
        }
    }

    if (outArea) *outArea = maxArea;
    return best;
}
