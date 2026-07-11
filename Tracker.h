#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QObject>
#include <QThread>
#include <QMetaType>
#include <limits>
#include <cmath>

Q_DECLARE_METATYPE(cv::Mat)

class Tracker : public QObject
{
    Q_OBJECT

private:
    int step;
    int N;
    int windowSize;
    float shiTomasiThreshold;
    int minPointsToRedetect;
    int nmsRadius;

    static constexpr int kBaseStep = 5;
    static constexpr int kBaseWindowSize = 7;
    static constexpr int kBaseNmsRadius = 2;
    static constexpr int kBaseMinPointsToRedetect = 30;

    void recalcStepDependentParams();
    
    // Детекция
    int diffThreshold = 22;
    int minObjectArea = 450;     
    int openKernelSize = 3;
    int closeKernelSize = 12;

    // Отслеживание объекта
    cv::Rect lastObjectBBox;
    bool hasLastObject = false;
    int missedFrames = 0;
    int maxMissedFrames = 15;
    float centerBiasRadiusRatio = 0.32f;

    float ransacReprojThreshold = 3.0f;

    // Anchor
    std::vector<cv::Point2f> anchorCorners;
    int framesSinceAnchor = 0;
    int anchorRefreshInterval = 10;
    int minAnchorBaselineFrames = 4;
    float anchorBaseReprojThreshold = 3.2f;

    // Motion Evidence
    cv::Mat motionEvidence;
    float motionEvidenceAlpha = 0.20f;
    float motionEvidenceThreshold = 0.4f;

    std::vector<cv::Point2f> trackedCorners;
    std::vector<uchar> pointStatus;

    cv::Mat prevGray;
    cv::Mat prevFrame;
    cv::Mat H;

    float centralZoneRatio = 0.68f;
    float classificationZoneRatio = 0.88f;
    int warpBorderErodePx = 10;

    // Радиальный буст
    bool useRadialBoost = true;
    float radialMotionThreshold = 0.57f;
    float radialBoostMaxCameraMotion = 14.0f;
    float minRadialMotionLen = 0.28f;

    int shadowPatchRadius = 3;
    float shadowValueRatioMin = 0.25f;   // темнее этого - не тень, а что-то другое
    float shadowValueRatioMax = 0.90f;   // светлее этого - разницы почти нет, не тень
    float shadowSatDeltaMax = 40.0f;     // макс. допустимое изменение насыщенности (0..255)
    float shadowHueDeltaMaxDeg = 20.0f;  // макс. допустимое изменение тона, в градусах

    double lastKnownArea = 0.0;

    std::vector<cv::Point2f> detectCorners(const cv::Mat& V);
    std::vector<float> buildIntegralImage(const std::vector<float>& data, int width, int height); 

    #ifdef QT_DEBUG
    // Отрисовка кадра: зелёные точки - фон, красные - объект.
    // Существует только в debug-сборке - в release не нужна для работы
    // алгоритма, только для визуальной отладки.
    cv::Mat drawVisualization(const cv::Mat& frame,
                               const std::vector<cv::Point2f>& backgroundPts,
                               const std::vector<cv::Point2f>& objectPts) const;
    #endif
    
    cv::Rect detectMovingObjectBBox(const cv::Mat& diffFrame, 
                                    const std::vector<cv::Point2f>& objectFeaturePts,
                                    int minArea = 500);

    bool isInCentralZone(const cv::Point2f& p, const cv::Size& frameSize, float zoneRatio) const;
    bool isNearLastObject(const cv::Point2f& p, float margin = 60.0f) const;
    bool isShadowPoint(const cv::Point2f& p, const cv::Mat& frameHSV, const cv::Mat& refHSV) const;

public:
    Tracker();    
    ~Tracker() = default;

    void setDetectionParams(int diffThreshold_, int minObjectArea_,
                             int openKernelSize_ = 3, int closeKernelSize_ = 15);

    void setGridStep(int step_);
    int getGridStep() const { return step; }
    void setRadialBoost(bool enable) { useRadialBoost = enable; }

public slots:
    void getFrame(const cv::Mat frame);

signals:
    void frameProcessed(cv::Mat visFrame, cv::Mat diffFrame);
};

#endif // TRACKER_HPP