#include "mainwindow.h"
#include <opencv2/core.hpp>
#include <QDebug>
#include <QApplication>
#include <QDebug>


int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "Russian");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return QApplication::exec();

}
