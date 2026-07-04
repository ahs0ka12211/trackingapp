#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QObject>
#include <QThread>
#include <mutex>


struct Claster {
    int R;
    int G;
    int B;

    int left;
    int right;
    int up;
    int down;

    std::vector<int> gridPoints; // пиксели из сетки принадлежащие к этому кластеру

    Claster();
};

class Tracker : public QObject
{

    Q_OBJECT

private:
    std::vector<std::array<int, 5>> gridEven; // [x, y, R, G, B]  
    std::vector<std::array<int, 5>> gridUnEven;

    std::vector<Claster> clastersEven; // кластеры объектов на кадре [средний цвет, крайняя левая, правая, верхняя, нижняя координаты]
    std::vector<Claster> clastersUnEven;

    int step; // шаг с которым строится сетка особых точек
    int N; // номер кадра 

    cv::VideoCapture* video;


public:
    Tracker();    

    ~Tracker();

    void DebugShowClusters(cv::Mat frame);

    void drawCluster(cv::Mat& overlay, 
                          const Claster& cl, 
                          const std::vector<std::array<int, 5>>& grid,
                          cv::Scalar color,
                          int index);

    void FindObject();

public slots:
    void getFrame(cv::Mat frame);
};
 
 
#endif // TRACKER_HPP