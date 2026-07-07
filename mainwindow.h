#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include "Tracker.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openVideo();
    void PauseVideo();
    void updateFrame();
    void onSliderMoved(int value);
    void onLockToggled();

private:
    QPushButton *btnOpen;
    QPushButton *btnPause;
    QPushButton *btnLock;   // кнопка фиксации на объекте
    QLabel      *videoLabel;
    QSlider     *videoSlider;
    QTimer      *timer;

    cv::VideoCapture cap;
    double totalFrames = 0;
    double fps         = 30;
    bool   isSliderUpdating = false;

    Tracker *tracker = nullptr;
};

#endif // MAINWINDOW_H
