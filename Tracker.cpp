#include "Tracker.h"
#include <QDebug>

Tracker::Tracker(cv::VideoCapture& cap) : _cap(cap)
{
    _mog2 = cv::createBackgroundSubtractorMOG2(200, 50, false);
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
    _mog2 = cv::createBackgroundSubtractorMOG2(200, 50, false);
    _hist.release();
    _detectionCount  = 0;
    _accumulatedMask = cv::Mat();
    _detecting       = true;
    _initialized     = false;
    _bbox            = cv::Rect();
    _start           = cv::Point();
    _end             = cv::Point();
}

// ─────────────────────────────────────────────────────────────
//  Фаза 1 — детекция объекта через MOG2
// ─────────────────────────────────────────────────────────────

bool Tracker::detectPhase(const cv::Mat& frame)
{
    cv::Mat mask;
    _mog2->apply(frame, mask);

    // Убираем шум морфологией
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Накапливаем маску за все кадры детекции
    if (_accumulatedMask.empty())
        _accumulatedMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::bitwise_or(_accumulatedMask, mask, _accumulatedMask);

    _detectionCount++;
    if (_detectionCount < DETECTION_FRAMES)
        return false;

    // Анализируем накопленную маску
    cv::Rect detected = largestContourRect(_accumulatedMask);

    if (detected.empty()) {
        qDebug() << "[Tracker] объект не обнаружен, повторяем детекцию";
        _detectionCount  = 0;
        _accumulatedMask = cv::Mat();
        return false;
    }

    _bbox = detected;

    // Строим гистограмму объекта для CamShift
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
//  Фаза 2 — трекинг через CamShift
// ─────────────────────────────────────────────────────────────

bool Tracker::trackPhase(const cv::Mat& frame)
{
    // Переводим кадр в HSV
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // Маска: убираем слишком тёмные и слишком светлые пиксели
    cv::Mat rangeMask;
    cv::inRange(hsv, cv::Scalar(0, 30, 30), cv::Scalar(180, 255, 255), rangeMask);

    // Back-projection: насколько каждый пиксель похож на объект
    cv::Mat backProj;
    int     channels[] = {0};
    float   hRange[]   = {0, 180};
    const float* ranges[] = {hRange};
    cv::calcBackProject(&hsv, 1, channels, _hist, backProj, ranges);
    cv::bitwise_and(backProj, rangeMask, backProj);

    // CamShift — адаптивно меняет и позицию, и размер прямоугольника
    cv::TermCriteria termCrit(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
        CAMSHIFT_ITERATIONS,
        CAMSHIFT_EPSILON
    );

    cv::RotatedRect rotRect = cv::CamShift(backProj, _bbox, termCrit);
    _bbox = rotRect.boundingRect();

    // Проверяем, что bbox не вышел за пределы кадра
    cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    _bbox &= frameRect; // пересечение

    if (_bbox.area() < MIN_CONTOUR_AREA) {
        qDebug() << "[Tracker] объект потерян (bbox слишком мал), возврат к детекции";
        reset();
        return false;
    }

    _start = _bbox.tl();
    _end   = _bbox.br();
    return true;
}

// ─────────────────────────────────────────────────────────────
//  Строим HSV-гистограмму объекта по начальному bbox
// ─────────────────────────────────────────────────────────────

void Tracker::buildHistogram(const cv::Mat& frame, const cv::Rect& roi)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat roiHsv = hsv(roi);

    // Маска: только "живые" цвета (не слишком тёмные/светлые)
    cv::Mat mask;
    cv::inRange(roiHsv, cv::Scalar(0, 30, 30), cv::Scalar(180, 255, 255), mask);

    // Строим 1D гистограмму по каналу Hue (0..180)
    int     histSize   = 64;
    float   hRange[]   = {0, 180};
    const float* ranges[] = {hRange};
    int     channels[] = {0};

    cv::calcHist(&roiHsv, 1, channels, mask, _hist, 1, &histSize, ranges);
    cv::normalize(_hist, _hist, 0, 255, cv::NORM_MINMAX);
}

// ─────────────────────────────────────────────────────────────
//  Самый крупный контур на накопленной маске
// ─────────────────────────────────────────────────────────────

cv::Rect Tracker::largestContourRect(const cv::Mat& mask) const
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double   maxArea = 0;
    cv::Rect best;

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area > maxArea && area >= MIN_CONTOUR_AREA) {
            maxArea = area;
            best    = cv::boundingRect(c);
        }
    }
    return best;
}
