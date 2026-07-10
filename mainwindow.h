#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QThread>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <opencv2/opencv.hpp>
#include "Tracker.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void openVideo();
    void PauseVideo();
    void updateFrame();
    void onSliderMoved(int value);
    void StartTracker();
    void onFrameProcessed(cv::Mat visFrame, cv::Mat diffFrame);

signals:
    void sendFrame(cv::Mat frame);  // Сигнал для отправки кадра в трекер

private:
    void showFrame(const cv::Mat& frame);
    void closeEvent(QCloseEvent *event) override;

private:
    cv::VideoCapture cap;
    
    QLabel *videoLabel;
    QLabel *diffLabel;
    QPushButton *btnOpen;
    QPushButton *btnPause;
    QPushButton *btnStartTracker;
    QSlider *videoSlider;
    QTimer *timer;
    
    QThread *trackerThread;
    Tracker *tracker;
    
    double fps;
    int totalFrames;
    bool isTracking;
    bool isSliderUpdating;
};

#endif // MAINWINDOW_H