## **ПРОЕКТ - ОТСЛЕЖИВАНИЕ ДВИЖУЩИХСЯ ОБЪЕКТОВ ПО ВИДЕО ПОТОКУ**

Проект реализован на  `C++` с использованием `QT` и `OpenCV`

## **Это версия (master ветка) проекта для отслеживания объектов с помощью всторенных средств `QT` и `OpenCV` (через `MOG2`)**

Для сборки проекта требуется в **`.pro`** файле заменить **`OPENCV_DIR`** на вашу директорию с `OpenCV`
```
OPENCV_DIR = Путь_к_OpenCV\x64\mingw
```
Для конфигурации проекта в **`vs code`** вы можете использовать следующие `.json` файлы в папке `.vscode`:

## Файл **`c_cpp_properties.json`**
В `includePath` добавьте соответствующие пути к библиотекам `QT`, в `compilerPath` укажите свой компилятор (не обязательно из `QT`)
```
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "D:/UCHEBA/RADAR/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/include",
                "D:/UCHEBA/RADAR/QT/6.11.1/mingw_64/include",
                "D:/UCHEBA/RADAR/QT/6.11.1/mingw_64/include/QtCore",
                "D:/UCHEBA/RADAR/QT/6.11.1/mingw_64/include/QtGui",
                "D:/UCHEBA/RADAR/QT/6.11.1/mingw_64/include/QtWidgets"
            ],
            "defines": [
                "UNICODE",
                "_UNICODE"
            ],
            "compilerPath": "D:/UCHEBA/PROG/mingw64/bin/g++.exe",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-gcc-x64",
            "compilerArgs": [
                "-std=gnu++1z"
            ]
        }
    ],
    "version": 4
}
```
## Файл **`settings.json`**
В `PATH` добавьте соответствующий путь к `QT`
```
{
    "terminal.integrated.env.windows": {
        "PATH": "D:/UCHEBA/RADAR/QT/6.11.1/mingw_64/bin;${env:PATH}"
    }
}
```
## Файл **`tasks.json`**
Если вы переименовывали `.pro` файл, то не забудьте изменить путь к нему, а также имя `.exe` на такое же, как и у `.pro`
```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Create build dir",
            "type": "shell",
            "command": "mkdir -Force build/"
        },
        {
            "label": "Run qmake",
            "type": "shell",
            "command": "qmake",
            "args": ["../trackingapp.pro"], // путь до .pro файла
            "options": {
                "cwd": "build/"
            }
        },
        {
            "label": "Run make",
            "type": "shell",
            "command": "mingw32-make", // или jom/nmake
            "options": {
                "cwd": "build/"
            }
        },
        {
            "label": "Run make debug",
            "type": "shell",
            "command": "mingw32-make debug", // или jom/nmake
            "options": {
                "cwd": "build/"
            }
        },
        {
            "label": "Build",
            "dependsOn": ["Create build dir", "Run qmake", "Run make", "Run make debug"],
            "dependsOrder": "sequence",
            "group": {
                "kind": "build",
                "isDefault": true // чтобы собирать по Ctrl+Shift+B
            }
        },
        {
            "label": "Run Project debug",
            "type": "shell",
            "command": "${workspaceFolder}/build/debug/trackingapp.exe", //имя .exe совпадает с именем .pro файла
            "options": {
                "cwd": "${workspaceFolder}/build/debug"
            },
            "group": "none",
            "problemMatcher": []
        },
        {
            "label": "Run Project",
            "type": "shell",
            "command": "${workspaceFolder}/build/release/trackingapp.exe", //имя .exe совпадает с именем .pro файла
            "options": {
                "cwd": "${workspaceFolder}/build/release"
            },
            "group": "none",
            "problemMatcher": []
        }
    ]
}
```
## Важно
Также для запуска проекта в папке с `.exe` должны находится соответствующие **`.dll`** для `OpenCV`
(обычно это папки `release` и `debug`)