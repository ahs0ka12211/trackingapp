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

    // Кнопка фиксации — работает как переключатель
    btnLock = new QPushButton("Зафиксировать объект", this);
    btnLock->setEnabled(false); // недоступна пока трекер не нашёл объект
    btnLock->setCheckable(true);
    btnLock->setStyleSheet(
        "QPushButton { background-color: #444; color: white; }"
        "QPushButton:checked { background-color: #c0392b; color: white; }"
    );
    layout->addWidget(btnLock);

    setCentralWidget(central);

    timer = new QTimer(this);
    connect(timer,       &QTimer::timeout,        this, &MainWindow::updateFrame);
    connect(btnOpen,     &QPushButton::clicked,   this, &MainWindow::openVideo);
    connect(btnPause,    &QPushButton::clicked,   this, &MainWindow::PauseVideo);
    connect(btnLock,     &QPushButton::clicked,   this, &MainWindow::onLockToggled);
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

    delete tracker;
    tracker = new Tracker(cap);

    // Сбрасываем кнопку при открытии нового видео
    btnLock->setChecked(false);
    btnLock->setEnabled(false);
    btnLock->setText("Зафиксировать объект");

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

void MainWindow::onLockToggled()
{
    if (!tracker) return;

    bool locked = btnLock->isChecked();
    tracker->setLocked(locked);
    btnLock->setText(locked ? "Снять фиксацию" : "Зафиксировать объект");

    qDebug() << (locked ? "[UI] объект зафиксирован" : "[UI] фиксация снята");
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

    isSliderUpdating = true;
    videoSlider->setValue(static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)));
    isSliderUpdating = false;

    if (tracker) {
        bool found = tracker->findObject(frame);

        if (tracker->isDetecting()) {
            // Во время детекции кнопка недоступна
            btnLock->setEnabled(false);
            btnLock->setChecked(false);
            btnLock->setText("Зафиксировать объект");

            cv::putText(frame, "Detecting object...",
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 200, 255), 2);
        }
        else if (found) {
            // Объект найден — разрешаем кнопку
            btnLock->setEnabled(true);

            cv::Scalar color = tracker->isLocked()
                ? cv::Scalar(0, 0, 255)   // красный — зафиксирован
                : cv::Scalar(0, 255, 0);  // зелёный — обычный трекинг

            cv::rectangle(frame,
                          tracker->get_start(),
                          tracker->get_end(),
                          color, 2);

            std::string label = tracker->isLocked() ? "Locked" : "Tracking";
            cv::putText(frame, label,
                        tracker->get_start() + cv::Point(0, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2);
        }
        else {
            // Объект потерян — сбрасываем фиксацию и кнопку
            btnLock->setEnabled(false);
            btnLock->setChecked(false);
            btnLock->setText("Зафиксировать объект");

            cv::putText(frame, "Object lost, re-detecting...",
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
        }
    }

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

    if (tracker) {
        tracker->reset(); // reset() снимает фиксацию внутри
        btnLock->setEnabled(false);
        btnLock->setChecked(false);
        btnLock->setText("Зафиксировать объект");
    }

    updateFrame();
}
