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
#include <limits>

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
    int openKernelSize = 3;      // ядро MORPH_OPEN - убирает шум, но может стереть мелкий/тонкий объект
    int closeKernelSize = 15;    // ядро MORPH_CLOSE - склеивает силуэт объекта

    // "привязка" к найденному объекту между кадрами: при параллаксе (едущая
    // вперёд камера) ближние объекты фона (деревья, столбы у обочины) часто
    // дают diff-пятно БОЛЬШЕ, чем реальная цель - брать "самый большой контур"
    // в таких сценах неверно. Вместо этого после первого захвата ищем контур,
    // ближайший по положению к прошлому кадру.
    cv::Rect lastObjectBBox;
    bool hasLastObject = false;
    int missedFrames = 0;
    int maxMissedFrames = 10;         // столько кадров подряд ищем рядом с прошлым положением,
                                       // прежде чем сбросить привязку и начать поиск заново
    float centerBiasRadiusRatio = 0.35f; // на холодном старте отдаём предпочтение контурам
                                          // ближе к центру кадра (объект съёмки обычно в центре,
                                          // а не у края, где чаще всего обочина/параллакс-шум)

    float ransacReprojThreshold;                 // порог репроекции (в пикселях) для RANSAC при поиске гомографии.
                                                  // Чем больше - тем терпимее к шуму, но тем легче объект "спутать" с фоном

    std::vector<cv::Point2f> trackedCorners;     // точки, которые трекаем оптическим потоком между кадрами
    std::vector<uchar> pointStatus;              // 1 = точка фона (инлаер гомографии), 0 = точка объекта (аутлаер) - классификация одного кадра

    std::vector<float> pointConfidence;          // персистентная (накопленная за много кадров) уверенность
                                                  // каждой ОТСЛЕЖИВАЕМОЙ ТОЧКИ (по identity через LK, а не по
                                                  // пикселю кадра) в том, что это объект, а не фон/шум.
                                                  // 0 = точно фон, 1 = точно объект. Копится по формуле
                                                  // EMA: conf = alpha*instant + (1-alpha)*conf_prev, поэтому
                                                  // случайный шум на один кадр не успевает поднять уверенность,
                                                  // а медленный/слабый объект копит её за счёт persistent identity
                                                  // точки (в отличие от diff по пикселям, где сигнал "уезжает"
                                                  // вместе с объектом и никогда не накапливается в одном месте)
    float confidenceAlpha = 0.15f;      // скорость обновления уверенности: меньше -> устойчивее к шуму
                                         // на один кадр, но дольше "разгоняется" и дольше "остывает"
    float confidenceThreshold = 0.55f;  // порог уверенности, начиная с которого точка считается объектом
    int minObjectPoints = 3;            // минимум уверенных точек в кластере, чтобы не приняли шум за объект
    float clusterRadius = 80.0f;        // макс. расстояние (px) между точками одного кластера-объекта

    cv::Mat prevGray;                            // V-канал предыдущего кадра (нужен для optical flow)
    cv::Mat prevFrame;                           // предыдущий кадр целиком BGR (нужен для варпинга/разности)
    cv::Mat H;                                   // последняя посчитанная гомография: prevFrame -> frame

    cv::Mat diffAccumulator;   // накопленный (экспоненциальное скользящее среднее) diffFrame,
                                // CV_32F - усиливает сигнал медленных/слабых объектов, не трогая H
    float diffAccumAlpha = 0.3f; // вес нового кадра в накоплении: больше -> быстрее реакция,
                                  // но меньше подавление шума; меньше -> лучше видно медленные
                                  // объекты, но объект "смазывается" по времени (инерция реакции)

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

    // Строит bbox объекта по кластеру точек с накопленной уверенностью выше
    // confidenceThreshold - основной (более устойчивый к шуму и медленным
    // объектам) источник детекции. Возвращает cv::Rect(), если подходящего
    // кластера нет (напр. объект совсем без текстуры/углов).
    cv::Rect clusterObjectBBox(const std::vector<cv::Point2f>& confidentObjectPts) const;

    bool isInCentralZone(const cv::Point2f& p, const cv::Size& frameSize, float zoneRatio) const;

public:
    Tracker();    

    ~Tracker() = default;

    // Ручная подстройка под тип сцены без пересборки:
    // - мелкий/тусклый объект на однородном фоне (напр. самолёт в небе) -
    //   уменьшить diffThreshold, minObjectArea, openKernelSize
    // - крупная цель среди параллакса (напр. обгон на дороге) - обычно
    //   подходят дефолты, но minObjectArea можно поднять, чтобы не хватать
    //   мелкий шум от травы/веток на обочине
    void setDetectionParams(int diffThreshold_, int minObjectArea_,
                             int openKernelSize_ = 3, int closeKernelSize_ = 15);
 
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