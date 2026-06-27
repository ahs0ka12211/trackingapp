QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
# OpenCV 4.5.5 MinGW
OPENCV_DIR = W:/RADAR/OPENCV/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw

INCLUDEPATH += $$OPENCV_DIR/../../include

LIBS += -L$$OPENCV_DIR/lib \
        -lopencv_core455 \
        -lopencv_highgui455 \
        -lopencv_imgproc455 \
        -lopencv_imgcodecs455 \
        -lopencv_videoio455 \
        -lopencv_dnn455 \
        -lopencv_features2d455 \
        -lopencv_objdetect455 \
        -lopencv_ml455        \
        -lopencv_video455