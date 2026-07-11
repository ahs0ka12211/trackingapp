#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QObject>
#include <QThread>
#include <QMetaType>
#include <mutex>

// Регистрируем cv::Mat как Qt-метатип - без этого сигнал frameProcessed()
// нельзя безопасно доставить через QueuedConnection (например, если
// Tracker живёт в отдельном QThread, а слушатель - в GUI-потоке)
Q_DECLARE_METATYPE(cv::Mat)

class Tracker : public QObject
{

    Q_OBJECT

private:
    int step;                                   // шаг с которым строится сетка особых точек
    int N;                                      // номер кадра 
    int windowSize;                             // размер окна блюра
    float shiTomasiThreshold;                   // коэфицент для отбора точек углов
    int minPointsToRedetect;                    // порог, ниже которого пере-детектируем углы
    int nmsRadius;                              // радиус подавления немаксимумов, в клетках сетки
    
    // параметры детекции объекта по diff
    int diffThreshold = 25;      // порог бинаризации разностного кадра
    int minObjectArea = 500;     // минимальная площадь контура, чтобы не ловить шум

    float ransacReprojThreshold;                 // порог репроекции (в пикселях) для RANSAC при поиске гомографии.
                                                  // Чем больше - тем терпимее к шуму, но тем легче объект "спутать" с фоном

    std::vector<cv::Point2f> trackedCorners;     // точки, которые трекаем оптическим потоком между кадрами
    std::vector<uchar> pointStatus;              // 1 = точка фона (инлаер гомографии), 0 = точка объекта (аутлаер)

    cv::Mat prevGray;                            // V-канал предыдущего кадра (нужен для optical flow)
    cv::Mat prevFrame;                           // предыдущий кадр целиком BGR (нужен для варпинга/разности)
    cv::Mat H;                                   // последняя посчитанная гомография: prevFrame -> frame

    float centralZoneRatio;   // доля кадра (по каждой оси), которая используется для оценки H

    float classificationZoneRatio;  // доля кадра, точки внутри которой можно классифицировать
                                     // как "объект"; точки за её пределами (у самой рамки)
                                     // всегда считаются фоном - там reprojection error
                                     // естественно выше из-за дисторсии/параллакса,
                                     // и жёсткий порог ransacReprojThreshold ошибочно
                                     // помечает их как объект

    int warpBorderErodePx;    // на сколько пикселей "сжимать" маску валидности warpPerspective,
                               // чтобы не ловить полупрозрачные краевые артефакты интерполяции

    std::vector<cv::Point2f> detectCorners(const cv::Mat& frame); 
    std::vector<float> buildIntegralImage(const std::vector<float>& data, int width, int height); 

    // Отрисовка кадра: зелёные точки - фон, красные - объект
    cv::Mat drawVisualization(const cv::Mat& frame,
                               const std::vector<cv::Point2f>& backgroundPts,
                               const std::vector<cv::Point2f>& objectPts) const;
    
    cv::Rect detectMovingObjectBBox(const cv::Mat& diffFrame, int minArea = 500);

    bool isInCentralZone(const cv::Point2f& p, const cv::Size& frameSize, float zoneRatio) const;

public:
    Tracker();    

    ~Tracker() = default;
 
public slots:
    
    void getFrame(const cv::Mat frame);

signals:

    // visFrame  - кадр с отрисованными точками (зелёные - фон, красные - объект)
    // diffFrame - разность между текущим кадром и предыдущим, "перемотанным"
    //             через гомографию H в текущий ракурс. В идеале фон тёмный,
    //             а объект, не подчиняющийся H, остаётся ярким пятном
    void frameProcessed(cv::Mat visFrame, cv::Mat diffFrame);
};
 
 
#endif // TRACKER_HPP