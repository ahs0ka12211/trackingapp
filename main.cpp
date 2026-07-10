#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Регистрируем cv::Mat для передачи между потоками через сигналы/слоты
    qRegisterMetaType<cv::Mat>("cv::Mat");
    
    MainWindow w;
    w.show();
    
    return app.exec();
}