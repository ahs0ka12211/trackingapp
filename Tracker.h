#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QObject>
#include <QMetaType>
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
    static constexpr int kBaseMinPointsToRedetect = 35;  // чуть повысили

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
    int maxMissedFrames = 18;           // чуть увеличил терпимость
    float centerBiasRadiusRatio = 0.32f;

    float ransacReprojThreshold = 3.0f;

    // Anchor
    std::vector<cv::Point2f> anchorCorners;
    int framesSinceAnchor = 0;
    int anchorRefreshInterval = 10;
    int minAnchorBaselineFrames = 4;
    float anchorBaseReprojThreshold = 3.3f;

    // Motion Evidence
    cv::Mat motionEvidence;
    float motionEvidenceAlpha = 0.22f;
    float motionEvidenceThreshold = 0.38f;

    // Фильтрация теней
    float shadowVRatioMin = 0.25f;
    float shadowVRatioMax = 0.92f;
    int   shadowSatDiffMax = 60;
    int   shadowHueDiffMax = 20;
    int   shadowOpenKernelSize = 3;

    std::vector<cv::Point2f> trackedCorners;
    std::vector<uchar> pointStatus;

    cv::Mat prevGray;
    cv::Mat prevFrame;
    cv::Mat H;

    float centralZoneRatio = 0.68f;
    float classificationZoneRatio = 0.88f;
    int warpBorderErodePx = 12;         // чуть увеличил

    // Радиальный буст
    bool useRadialBoost = true;
    float radialMotionThreshold = 0.57f;
    float radialBoostMaxCameraMotion = 15.0f;
    float minRadialMotionLen = 0.25f;   // чуть ослабил

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
    bool isNearLastObject(const cv::Point2f& p, float margin = 50.0f) const;

    cv::Rect warpBBox(const cv::Rect& bbox, const cv::Mat& H, const cv::Size& frameSize) const;
    cv::Mat computeShadowMask(const cv::Mat& currFrame, const cv::Mat& bgFrame) const;
    bool isShadowPoint(const cv::Point2f& p, const cv::Mat& frameHSV, const cv::Mat& refHSV) const;

public:
    Tracker();    
    ~Tracker() = default;

    void setDetectionParams(int diffThreshold_, int minObjectArea_,
                             int openKernelSize_ = 3, int closeKernelSize_ = 15);

    void setGridStep(int step_);
    int getGridStep() const { return step; }
    void setRadialBoost(bool enable) { useRadialBoost = enable; }

    void setShadowFilterParams(float vRatioMin, float vRatioMax,
                                int satDiffMax, int hueDiffMax)
    {
        shadowVRatioMin = vRatioMin;
        shadowVRatioMax = vRatioMax;
        shadowSatDiffMax = satDiffMax;
        shadowHueDiffMax = hueDiffMax;
    }

public slots:
    void getFrame(const cv::Mat frame);

signals:
    void frameProcessed(cv::Mat visFrame, cv::Mat diffFrame);
};

#endif // TRACKER_HPP