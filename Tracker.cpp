#include "Tracker.h"

Claster::Claster() 
{
    R=0;G=0;B=0;left=0;right=0;up=0;down=0;
}

Tracker::Tracker() : _gridEven(), _gridUnEven()
{ step = 5; N = 1;}

void Tracker::getFrame(cv::Mat frame){
    std::vector<std::array<int, 5>>* gridCur;
    std::vector<Claster>* clasters;
    if(N % 2 == 1){  // нечетный кадр
        gridCur = &_gridUnEven;
        clasters = &_clastersEven;
    }
    else {
        gridCur = &_gridEven;
        clasters = &_clastersUnEven;
    }
    gridCur->clear();
    clasters->clear();

    int cols = (frame.cols + step - 1) / step;
    int rows = (frame.size().height + step - 1) / step;

    // ==========================================================
    // Проход 1: просто сэмплируем сетку, без кластеризации.
    // _gridIndexBuf[gy*cols+gx] -> индекс точки в gridCur
    // ==========================================================
    size_t gridSize = static_cast<size_t>(rows) * cols;
    if (_gridIndexBuf.size() != gridSize)
        _gridIndexBuf.resize(gridSize);
    std::fill(_gridIndexBuf.begin(), _gridIndexBuf.end(), -1);

    _bfsQueue.resize(static_cast<size_t>(rows) * cols); // с запасом, максимум - все ячейки

    for (int y = 0, gy = 0; y < frame.size().height; y += step, gy++)
    {
        cv::Vec3b* row = frame.ptr<cv::Vec3b>(y);
        for (int x = 0, gx = 0; x < frame.cols; x += step, gx++) {
            int R = (int)row[x][2];
            int G = (int)row[x][1];
            int B = (int)row[x][0];
            gridCur->push_back(std::array<int, 5>{x, y, R, G, B});
            _gridIndexBuf[gy * cols + gx] = static_cast<int>(gridCur->size() - 1);
        }
    }

    // ==========================================================
    // Проход 2: BFS по сетке (connected components)
    // ==========================================================
    if (_visitedBuf.size() != gridSize)
        _visitedBuf.resize(gridSize);
    std::fill(_visitedBuf.begin(), _visitedBuf.end(), false);
    int qHead = 0, qTail = 0;

    // 8-связность (диагонали тоже считаем соседями).
    // Если нужна строгая 4-связность - оставь только первые 4 пары.
    static const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static const int dy[8] = { 0, 0,-1, 1, -1,  1,-1, 1};

    for (int gy = 0; gy < rows; gy++) {
        for (int gx = 0; gx < cols; gx++) {
            size_t flat = static_cast<size_t>(gy) * cols + gx;
            if (_visitedBuf[flat]) continue;
            _visitedBuf[flat] = true;

            int seedIdx = _gridIndexBuf[flat];
            const auto& seedPoint = (*gridCur)[seedIdx];

            // ЯКОРНЫЙ цвет кластера - фиксируем раз и навсегда.
            // Никакого EMA/скользящего среднего -> дрейфа быть не может.
            const int anchorR = seedPoint[2];
            const int anchorG = seedPoint[3];
            const int anchorB = seedPoint[4];

            Claster newCl;
            newCl.R = anchorR;
            newCl.G = anchorG;
            newCl.B = anchorB;
            newCl.left  = seedPoint[0];
            newCl.right = seedPoint[0];
            newCl.up    = seedPoint[1];
            newCl.down  = seedPoint[1];
            newCl.gridPoints.push_back(seedIdx);

            _bfsQueue[qTail++] = static_cast<int>(flat);

            while (qHead < qTail) {
                int curFlat = _bfsQueue[qHead++];
                int cx = curFlat % cols;
                int cy = curFlat / cols;

                for (int d = 0; d < 8; d++) {
                    int nx = cx + dx[d];
                    int ny = cy + dy[d];
                    if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;

                    size_t nFlat = static_cast<size_t>(ny) * cols + nx;
                    if (_visitedBuf[nFlat]) continue;

                    int pointIdx = _gridIndexBuf[nFlat];
                    const auto& p = (*gridCur)[pointIdx];
                    int R = p[2], G = p[3], B = p[4];

                    // Сравниваем с ЯКОРЕМ кластера, а не с текущим соседом -
                    // именно это убирает "уплывающий цвет".
                    if (abs(R - anchorR) < 30 && abs(G - anchorG) < 30 && abs(B - anchorB) < 30) {
                        _visitedBuf[nFlat] = true;
                        newCl.gridPoints.push_back(pointIdx);

                        int px = p[0], py = p[1];
                        if (px < newCl.left)  newCl.left  = px;
                        if (px > newCl.right) newCl.right = px;
                        if (py < newCl.up)    newCl.up    = py;
                        if (py > newCl.down)  newCl.down  = py;

                        _bfsQueue[qTail++] = static_cast<int>(nFlat);
                    }
                }
            }

            clasters->push_back(std::move(newCl));
        }
    }

    // Удаление шумовых кластеров
    for(int i = (*clasters).size() - 1; i >= 0; i--)
        if((*clasters)[i].gridPoints.size() < 5)
            (*clasters).erase((*clasters).begin() + i);

    N++;
    qDebug() << "grid size: " << gridCur->size();
    qDebug() << "clast size: " << (*clasters).size();

    static int debugCounter = 0;
    debugCounter++;
    if (debugCounter % 1 == 0) {
        debugShowClusters(frame);
    }

    if(gridCur->size() > 0)
        findObject();
}

void Tracker::findObject(){}

void Tracker::debugShowClusters(cv::Mat frame){
    // ==========================================
    // 1. Берем текущие кластеры
    // ==========================================
    std::vector<Claster>* clasters;
    std::vector<std::array<int, 5>>* grid;
    
    if (N % 2 == 1) {
        clasters = &_clastersEven;
        grid = &_gridUnEven;
    } else {
        clasters = &_clastersUnEven;
        grid = &_gridEven;
    }
    
    if (clasters->empty()) {
        qDebug() << "Нет кластеров для отображения!";
        return;
    }
    
    // ==========================================
    // 2. Создаем ОВЕРЛЕЙ (прозрачный слой)
    // ==========================================
    cv::Mat overlay = frame.clone(); // Копия кадра
    
    // ==========================================
    // 3. Генерируем случайные цвета для кластеров
    // ==========================================
    std::vector<cv::Scalar> colors;
    cv::RNG rng(12345);
    
    for (size_t i = 0; i < clasters->size(); i++) {
        int r = rng.uniform(100, 255);
        int g = rng.uniform(100, 255);
        int b = rng.uniform(100, 255);
        colors.push_back(cv::Scalar(b, g, r));
    }
    
    // ==========================================
    // 4. Рисуем КАЖДЫЙ кластер на оверлее
    // ==========================================
    for (size_t i = 0; i < clasters->size(); i++) {
        const Claster& cl = (*clasters)[i];
        cv::Scalar color = colors[i % colors.size()];
        
        // ---- Рисуем точки (прозрачные) ----
        for (int idx : cl.gridPoints) {
            int x = (*grid)[idx][0];
            int y = (*grid)[idx][1];
            // Рисуем точку с прозрачностью (через заливку с alpha)
            cv::circle(overlay, cv::Point(x, y), 3, color, -1); // -1 = заливка
        }
        
        // ---- Рисуем прямоугольник (контур) ----
        cv::rectangle(overlay, 
                      cv::Point(cl.left, cl.up), 
                      cv::Point(cl.right, cl.down), 
                      color, 2);
        
        // ---- Пишем номер и размер ----
        std::string label = std::to_string(i) + "(" + std::to_string(cl.gridPoints.size()) + ")";
        cv::putText(overlay, label, 
                    cv::Point(cl.left, cl.up - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    
    // ==========================================
    // 5. СМЕШИВАНИЕ (прозрачность)
    // ==========================================
    double alpha = 0.75; // 0.0 = полностью прозрачно, 1.0 = полностью непрозрачно
    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, overlay);
    
    // ==========================================
    // 6. МАСШТАБИРУЕМ (чтобы окно влезало на экран)
    // ==========================================
    cv::Mat displayFrame;
    double scale = 0.6; // 60% от оригинального размера
    cv::resize(overlay, displayFrame, cv::Size(), scale, scale);
    
    // ==========================================
    // 7. ПОКАЗЫВАЕМ
    // ==========================================
    cv::imshow("Clusters Debug", displayFrame);
    cv::waitKey(1);
    
    qDebug() << "Показано кластеров:" << clasters->size();
}