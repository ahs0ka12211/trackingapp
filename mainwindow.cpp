#include "mainwindow.h"
#include <QFileDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // Виджет и layout
    QWidget *central = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(central);

    // Кнопка открытия
    btnOpen = new QPushButton("Открыть видео", this);
    layout->addWidget(btnOpen);

    // Label для кадров
    videoLabel = new QLabel(this);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setMinimumSize(640, 480);
    videoLabel->setStyleSheet("background-color: white;");
    layout->addWidget(videoLabel);

    // ---- СЛАЙДЕР ----
    // Создаем горизонтальный слой для слайдера
    QHBoxLayout *sliderLayout = new QHBoxLayout();

    // Создаем слайдер
    videoSlider = new QSlider(Qt::Horizontal, this);
    videoSlider->setRange(0, 100); // Временный диапазон, обновится при открытии видео
    videoSlider->setEnabled(false); // Пока нет видео - выключен
    videoSlider->setToolTip("Перемотка видео");

    sliderLayout->addWidget(videoSlider);
    layout->addLayout(sliderLayout);
    // -----------------

    // Кнопка паузы
    btnPause = new QPushButton("Пауза/Стоп", this);
    layout->addWidget(btnPause);

    setCentralWidget(central);

    // Таймер для обновления кадров
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);
    connect(btnOpen, &QPushButton::clicked, this, &MainWindow::openVideo);
    connect(btnPause, &QPushButton::clicked, this, &MainWindow::PauseVideo);
    connect(videoSlider, &QSlider::sliderMoved, this, &MainWindow::onSliderMoved);
}

MainWindow::~MainWindow()
{
    cap.release();
}

void MainWindow::openVideo()
{
    QString path = QFileDialog::getOpenFileName(
        this, "Открыть видео", "",
        "Видео файлы (*.mp4 *.avi *.mkv *.mov *.jpeg *.png)"
        );

    if (path.isEmpty()) {
        qDebug() << "Файл не выбран";
        return;
    }

    qDebug() << "Открываем:" << path;

    // Закрываем предыдущее видео
    if (cap.isOpened()) {
        cap.release();
        timer->stop();
    }

    cap.open(path.toStdString());

    qDebug() << "isOpened:" << cap.isOpened();

    if (!cap.isOpened()) {
        qDebug() << "Не удалось открыть видео!";
        return;
    }

    // Получаем параметры видео
    fps = cap.get(cv::CAP_PROP_FPS);
    totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    qDebug() << "FPS:" << fps;
    qDebug() << "Всего кадров:" << totalFrames;

    if (fps <= 0) fps = 30;

    // Настраиваем слайдер
    videoSlider->setRange(0, totalFrames);
    videoSlider->setEnabled(true);
    videoSlider->setValue(0);

    // Запускаем воспроизведение
    timer->start(1000 / fps);
}

void MainWindow::PauseVideo()
{
    if (timer->isActive()) {
        timer->stop();
    } else {
        // При возобновлении синхронизируем слайдер с текущим кадром
        int currentFrame = cap.get(cv::CAP_PROP_POS_FRAMES);
        videoSlider->setValue(currentFrame);
        timer->start(1000 / fps);
    }
}

void MainWindow::updateFrame()
{
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        timer->stop();
        qDebug() << "Видео закончилось";
        return;
    }

    // Обновляем слайдер без отправки сигнала
    isSliderUpdating = true;
    int currentFrame = cap.get(cv::CAP_PROP_POS_FRAMES);
    videoSlider->setValue(currentFrame);
    isSliderUpdating = false;

    // Рисуем желтый прямоугольник
    cv::rectangle(frame,
                  cv::Point(100, 100),
                  cv::Point(300, 250),
                  cv::Scalar(0, 255, 255),
                  2
                  );

    // Конвертируем и показываем
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(
        videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::onSliderMoved(int value)
{
    if (isSliderUpdating) return; // Предотвращаем рекурсию

    if (!cap.isOpened()) return;


    // Устанавливаем позицию в видео
    cap.set(cv::CAP_PROP_POS_FRAMES, value);

    // Показываем текущий кадр
    updateFrame();
}