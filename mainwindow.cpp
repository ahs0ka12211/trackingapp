#include "mainwindow.h"
#include <QFileDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    QWidget     *central = new QWidget(this);
    QVBoxLayout *layout  = new QVBoxLayout(central);

    btnOpen = new QPushButton("Открыть видео", this);
    layout->addWidget(btnOpen);

    videoLabel = new QLabel(this);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setMinimumSize(640, 480);
    videoLabel->setStyleSheet("background-color: black;");
    layout->addWidget(videoLabel);

    QHBoxLayout *sliderLayout = new QHBoxLayout();
    videoSlider = new QSlider(Qt::Horizontal, this);
    videoSlider->setRange(0, 100);
    videoSlider->setEnabled(false);
    sliderLayout->addWidget(videoSlider);
    layout->addLayout(sliderLayout);

    btnPause = new QPushButton("Пауза/Стоп", this);
    layout->addWidget(btnPause);

    setCentralWidget(central);

    timer = new QTimer(this);
    connect(timer,       &QTimer::timeout,        this, &MainWindow::updateFrame);
    connect(btnOpen,     &QPushButton::clicked,   this, &MainWindow::openVideo);
    connect(btnPause,    &QPushButton::clicked,   this, &MainWindow::PauseVideo);
    connect(videoSlider, &QSlider::sliderMoved,   this, &MainWindow::onSliderMoved);
}

MainWindow::~MainWindow()
{
    cap.release();
    delete tracker;
}

// ─────────────────────────────────────────────────────────────

void MainWindow::openVideo()
{
    QString path = QFileDialog::getOpenFileName(
        this, "Открыть видео", "",
        "Видео файлы (*.mp4 *.avi *.mkv *.mov)"
    );

    if (path.isEmpty()) return;

    if (cap.isOpened()) {
        cap.release();
        timer->stop();
    }

    cap.open(path.toStdString());
    if (!cap.isOpened()) {
        qDebug() << "Не удалось открыть видео!";
        return;
    }

    fps         = cap.get(cv::CAP_PROP_FPS);
    totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (fps <= 0) fps = 30;

    videoSlider->setRange(0, static_cast<int>(totalFrames));
    videoSlider->setEnabled(true);
    videoSlider->setValue(0);

    // Пересоздаём трекер для нового видео
    delete tracker;
    tracker = new Tracker(cap);

    timer->start(1000 / static_cast<int>(fps));
}

void MainWindow::PauseVideo()
{
    if (timer->isActive()) {
        timer->stop();
    } else {
        timer->start(1000 / static_cast<int>(fps));
    }
}

// ─────────────────────────────────────────────────────────────

void MainWindow::updateFrame()
{
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        timer->stop();
        qDebug() << "Видео закончилось";
        return;
    }

    // Обновляем слайдер
    isSliderUpdating = true;
    videoSlider->setValue(static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)));
    isSliderUpdating = false;

    // ── Трекинг ──────────────────────────────────────────────
    if (tracker) {
        bool found = tracker->findObject(frame);

        if (tracker->isDetecting()) {
            // Идёт фаза детекции — показываем статус
            cv::putText(frame,
                        "Detecting object...",
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 200, 255), 2);
        }
        else if (found) {
            // Трекинг активен — рисуем прямоугольник
            cv::rectangle(frame,
                          tracker->get_start(),
                          tracker->get_end(),
                          cv::Scalar(0, 255, 0),   // зелёный
                          2);

            // Подпись над прямоугольником
            cv::putText(frame,
                        "Tracking",
                        tracker->get_start() + cv::Point(0, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
        else {
            // Трекер потерял объект
            cv::putText(frame,
                        "Object lost, re-detecting...",
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
        }
    }
    // ─────────────────────────────────────────────────────────

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows,
                static_cast<int>(frame.step), QImage::Format_RGB888);
    videoLabel->setPixmap(
        QPixmap::fromImage(qimg).scaled(
            videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// ─────────────────────────────────────────────────────────────

void MainWindow::onSliderMoved(int value)
{
    if (isSliderUpdating || !cap.isOpened()) return;

    cap.set(cv::CAP_PROP_POS_FRAMES, value);

    // При перемотке сбрасываем трекер — нужно заново детектировать
    if (tracker) tracker->reset();

    updateFrame();
}
