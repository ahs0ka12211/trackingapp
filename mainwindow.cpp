#include "mainwindow.h"
#include <QFileDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QThread>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , cap()
    , videoLabel(nullptr)
    , diffLabel(nullptr)
    , btnOpen(nullptr)
    , btnPause(nullptr)
    , btnStartTracker(nullptr)
    , videoSlider(nullptr)
    , timer(nullptr)
    , trackerThread(nullptr)
    , tracker(nullptr)
    , fps(30.0)
    , totalFrames(0)
    , isTracking(false)
    , isSliderUpdating(false)
{
    // ==========================================================
    // 1. СОЗДАНИЕ UI
    // ==========================================================
    
    // Виджет и layout
    QWidget *central = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(central);

    // Кнопка открытия
    btnOpen = new QPushButton("Открыть видео", this);
    layout->addWidget(btnOpen);

    // Label для основного кадра
    videoLabel = new QLabel(this);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setMinimumSize(640, 480);
    videoLabel->setStyleSheet("background-color: black;");
    layout->addWidget(videoLabel);

    // Label для разностного кадра (diff)
    diffLabel = new QLabel(this);
    diffLabel->setAlignment(Qt::AlignCenter);
    diffLabel->setMinimumSize(320, 240);
    diffLabel->setStyleSheet("background-color: black;");
    layout->addWidget(diffLabel);

    // Слайдер
    QHBoxLayout *sliderLayout = new QHBoxLayout();
    videoSlider = new QSlider(Qt::Horizontal, this);
    videoSlider->setRange(0, 100);
    videoSlider->setEnabled(false);
    videoSlider->setToolTip("Перемотка видео");
    sliderLayout->addWidget(videoSlider);
    layout->addLayout(sliderLayout);

    // Кнопка паузы
    btnPause = new QPushButton("Пауза/Стоп", this);
    layout->addWidget(btnPause);

    // Кнопка запуска трекера
    btnStartTracker = new QPushButton("Запустить трекер", this);
    layout->addWidget(btnStartTracker);

    setCentralWidget(central);

    // ==========================================================
    // 2. СОЗДАНИЕ ТРЕКЕРА В ОТДЕЛЬНОМ ПОТОКЕ
    // ==========================================================
    
    trackerThread = new QThread(this);
    tracker = new Tracker();
    tracker->moveToThread(trackerThread);
    
    // Подключаем сигнал от трекера к слоту отображения
    connect(tracker, &Tracker::frameProcessed, 
            this, &MainWindow::onFrameProcessed);
    
    // Подключаем сигнал для отправки кадров в трекер
    connect(this, &MainWindow::sendFrame, 
            tracker, &Tracker::getFrame, Qt::QueuedConnection);
    
    // ПОДКЛЮЧАЕМ СИГНАЛ СБРОСА ТРЕКЕРА
    connect(this, &MainWindow::resetTrackerSignal,
            this, &MainWindow::resetTracker, Qt::QueuedConnection);
    
    // Запускаем поток трекера
    trackerThread->start();

    // ==========================================================
    // 3. ТАЙМЕР И КНОПКИ
    // ==========================================================
    
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);
    connect(btnOpen, &QPushButton::clicked, this, &MainWindow::openVideo);
    connect(btnPause, &QPushButton::clicked, this, &MainWindow::PauseVideo);
    connect(btnStartTracker, &QPushButton::clicked, this, &MainWindow::StartTracker);
    connect(videoSlider, &QSlider::sliderMoved, this, &MainWindow::onSliderMoved);
}

MainWindow::~MainWindow()
{
    // Останавливаем таймер
    if (timer && timer->isActive()) {
        timer->stop();
    }
    
    // Останавливаем поток трекера
    if (trackerThread && trackerThread->isRunning()) {
        trackerThread->quit();
        trackerThread->wait();
    }
    
    // Освобождаем видео
    if (cap.isOpened()) {
        cap.release();
    }
}

// ==========================================================
// 3.5. СБРОС ТРЕКЕРА (НОВЫЙ МЕТОД)
// ==========================================================

void MainWindow::resetTracker()
{
    if (!tracker) return;
    
    // Создаем новый трекер (старый удалится автоматически)
    // Но проще пересоздать объект в потоке
    
    // Останавливаем текущий трекер
    tracker->deleteLater();
    
    // Создаем новый
    tracker = new Tracker();
    tracker->moveToThread(trackerThread);
    
    // Переподключаем сигналы
    connect(tracker, &Tracker::frameProcessed, 
            this, &MainWindow::onFrameProcessed);
    connect(this, &MainWindow::sendFrame, 
            tracker, &Tracker::getFrame, Qt::QueuedConnection);
    
    qDebug() << "[MainWindow] Трекер сброшен до изначального состояния";
}

// ==========================================================
// 4. ОТКРЫТИЕ ВИДЕО
// ==========================================================

void MainWindow::openVideo()
{
    QString path = QFileDialog::getOpenFileName(
        this, "Открыть видео", "",
        "Видео файлы (*.mp4 *.avi *.mkv *.mov);;Изображения (*.jpeg *.png *.jpg)"
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

    // ==========================================================
    // СБРОС ТРЕКЕРА ПРИ ОТКРЫТИИ НОВОГО ВИДЕО
    // ==========================================================
    emit resetTrackerSignal();
    
    // Сбрасываем состояние трекера в UI
    isTracking = false;
    btnStartTracker->setText("Запустить трекер");
    
    // Очищаем diffLabel
    diffLabel->clear();

    cap.open(path.toStdString());

    qDebug() << "isOpened:" << cap.isOpened();

    if (!cap.isOpened()) {
        qDebug() << "Не удалось открыть видео!";
        return;
    }

    // Получаем параметры видео
    fps = cap.get(cv::CAP_PROP_FPS);
    totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    qDebug() << "FPS:" << fps;
    qDebug() << "Frame count:" << totalFrames;

    if (fps <= 0) fps = 30;

    // Настраиваем слайдер
    videoSlider->setRange(0, totalFrames);
    videoSlider->setEnabled(true);
    videoSlider->setValue(0);

    // Запускаем воспроизведение
    timer->start(1000 / fps);
}

// ==========================================================
// 5. УПРАВЛЕНИЕ ВОСПРОИЗВЕДЕНИЕМ
// ==========================================================

void MainWindow::PauseVideo()
{
    if (timer->isActive()) {
        timer->stop();
        btnPause->setText("Продолжить");
        
        // ==========================================================
        // СБРОС ТРЕКЕРА ПРИ ПАУЗЕ
        // ==========================================================
        emit resetTrackerSignal();
        
        // Сбрасываем состояние трекера в UI
        isTracking = false;
        btnStartTracker->setText("Запустить трекер");
        
        // Очищаем diffLabel
        diffLabel->clear();
        
    } else {
        // При возобновлении синхронизируем слайдер с текущим кадром
        int currentFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
        videoSlider->setValue(currentFrame);
        timer->start(1000 / fps);
        btnPause->setText("Пауза/Стоп");
    }
}

void MainWindow::StartTracker()
{
    isTracking = !isTracking;  // Переключаем состояние
    
    if (isTracking) {
        btnStartTracker->setText("Остановить трекер");
        qDebug() << "Трекер ЗАПУЩЕН";
    } else {
        btnStartTracker->setText("Запустить трекер");
        qDebug() << "Трекер ОСТАНОВЛЕН";
        
        // ==========================================================
        // СБРОС ТРЕКЕРА ПРИ ОСТАНОВКЕ
        // ==========================================================
        emit resetTrackerSignal();
        
        // Очищаем diffLabel
        diffLabel->clear();
    }
}

// ==========================================================
// 6. ОБНОВЛЕНИЕ КАДРА (ВЫЗЫВАЕТСЯ ПО ТАЙМЕРУ)
// ==========================================================

void MainWindow::updateFrame()
{
    if (!cap.isOpened()) {
        return;
    }
    
    cv::Mat frame;
    cap >> frame;
    
    if (frame.empty()) {
        timer->stop();
        qDebug() << "Видео закончилось";
        
        // ==========================================================
        // СБРОС ТРЕКЕРА ПРИ ЗАВЕРШЕНИИ ВИДЕО
        // ==========================================================
        emit resetTrackerSignal();
        isTracking = false;
        btnStartTracker->setText("Запустить трекер");
        diffLabel->clear();
        
        return;
    }
    
    // ==========================================================
    // ОТПРАВКА КАДРА В ТРЕКЕР (ЕСЛИ ВКЛЮЧЕН)
    // ==========================================================
    
    if (isTracking) {
        // Отправляем копию кадра в трекер (через сигнал, чтобы попасть в поток трекера)
        emit sendFrame(frame.clone());
    }
    
    // ==========================================================
    // ОТОБРАЖЕНИЕ КАДРА НА ЭКРАНЕ
    // ==========================================================
    
    // Если трекер выключен - показываем оригинальный кадр
    // Если трекер включен - кадр обновится через onFrameProcessed
    if (!isTracking) {
        showFrame(frame);
    }
    
    // ==========================================================
    // ОБНОВЛЕНИЕ СЛАЙДЕРА
    // ==========================================================
    
    isSliderUpdating = true;
    int currentFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
    videoSlider->setValue(currentFrame);
    isSliderUpdating = false;
}

// ==========================================================
// 7. ОТОБРАЖЕНИЕ КАДРА НА ЭКРАНЕ
// ==========================================================

void MainWindow::showFrame(const cv::Mat& frame)
{
    if (frame.empty()) return;
    
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(
        videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// ==========================================================
// 8. СЛОТ ДЛЯ ПРИЕМА РЕЗУЛЬТАТОВ ОТ ТРЕКЕРА
// ==========================================================

void MainWindow::onFrameProcessed(cv::Mat visFrame, cv::Mat diffFrame)
{
    // ==========================================================
    // ОТОБРАЖЕНИЕ ВИЗУАЛИЗАЦИИ (ЗЕЛЕНЫЕ/КРАСНЫЕ ТОЧКИ)
    // ==========================================================
    
    if (!visFrame.empty()) {
        cv::Mat rgb;
        cv::cvtColor(visFrame, rgb, cv::COLOR_BGR2RGB);
        QImage qimgVis(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        videoLabel->setPixmap(QPixmap::fromImage(qimgVis).scaled(
            videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    
    // ==========================================================
    // ОТОБРАЖЕНИЕ РАЗНОСТНОГО КАДРА (DIFF)
    // ==========================================================
    
    if (!diffFrame.empty()) {
        cv::Mat diffColor;
        
        // Если diffFrame одноканальный (grayscale) - конвертим в цветной
        if (diffFrame.channels() == 1) {
            cv::cvtColor(diffFrame, diffColor, cv::COLOR_GRAY2BGR);
        } else {
            diffColor = diffFrame.clone();
        }
        
        // Конвертим BGR -> RGB для Qt
        cv::Mat rgbDiff;
        cv::cvtColor(diffColor, rgbDiff, cv::COLOR_BGR2RGB);
        QImage qimgDiff(rgbDiff.data, rgbDiff.cols, rgbDiff.rows, 
                        rgbDiff.step, QImage::Format_RGB888);
        diffLabel->setPixmap(QPixmap::fromImage(qimgDiff).scaled(
            diffLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

// ==========================================================
// 9. СЛАЙДЕР ПЕРЕМОТКИ
// ==========================================================

void MainWindow::onSliderMoved(int value)
{
    if (isSliderUpdating) return;
    if (!cap.isOpened()) return;
    
    // ==========================================================
    // СБРОС ТРЕКЕРА ПРИ ПЕРЕМОТКЕ
    // ==========================================================
    emit resetTrackerSignal();
    
    // Сбрасываем состояние трекера в UI
    isTracking = false;
    btnStartTracker->setText("Запустить трекер");
    
    // Очищаем diffLabel
    diffLabel->clear();
    
    // Устанавливаем позицию в видео
    cap.set(cv::CAP_PROP_POS_FRAMES, value);
    
    // Показываем текущий кадр
    updateFrame();
}

// ==========================================================
// 10. ОБРАБОТКА ЗАКРЫТИЯ ОКНА
// ==========================================================

void MainWindow::closeEvent(QCloseEvent *event)
{
    // Останавливаем таймер
    if (timer && timer->isActive()) {
        timer->stop();
    }
    
    // Останавливаем поток трекера
    if (trackerThread && trackerThread->isRunning()) {
        trackerThread->quit();
        trackerThread->wait();
    }
    
    // Освобождаем видео
    if (cap.isOpened()) {
        cap.release();
    }
    
    QMainWindow::closeEvent(event);
}