#include "Tracker.h"

Tracker::Tracker()
{
    step = 5; N = 0;
    windowSize = 7;
    shiTomasiThreshold = 0.05f;          
    minPointsToRedetect = 30;                
    nmsRadius = 2;    
    ransacReprojThreshold = 3.0f;

    qRegisterMetaType<cv::Mat>("cv::Mat"); // чтобы frameProcessed() работал через QueuedConnection
} 
 

// Построение интегрального изображения
std::vector<float> Tracker::buildIntegralImage(const std::vector<float>& data, int width, int height) {

    // Дополнительно увеличиваем ширину и длину на +1 чтобы 
    // на границах сетки (x = 0 | y = 0) не было ошибок
    std::vector<float> integral((width + 1) * (height + 1), 0.0f);
    

    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++){
            float val = data[y*width + x];
            float left = integral[(y + 1) * (width + 1) + x];
            float up = integral[y * (width + 1) + (x + 1)];
            float leftup = integral[y * (width + 1) + x];
            integral[ (y + 1) * (width + 1) + (x + 1)] = val + left + up - leftup;
        }

    return integral;
}


void Tracker::GetFrame(const cv::Mat frame){
 
    cv::Mat frameHSV;
    cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(frameHSV, channels);
    cv::Mat V = channels[2];

    // ---------- Первый кадр: детектируем углы, запоминаем и выходим ----------
    // На первом кадре сравнивать ещё не с чем - гомографию считать не из чего
    if (N == 0) {
        trackedCorners = detectCorners(frame);
        pointStatus.assign(trackedCorners.size(), 1); // изначально считаем всё фоном
        prevGray = V.clone();
        prevFrame = frame.clone();
        H = cv::Mat::eye(3, 3, CV_64F);
        N++;
        return;
    }

    // ==========================================================
    // 1. Трекаем точки предыдущего кадра пирамидальным Lucas-Kanade
    //    оптическим потоком. Это даёт нам соответствия
    //    "точка на прошлом кадре" <-> "та же точка на текущем"
    // ==========================================================
    std::vector<cv::Point2f> nextPts;
    std::vector<uchar> lkStatus;
    std::vector<float> lkErr;

    if (!trackedCorners.empty()) {
        cv::calcOpticalFlowPyrLK(
            prevGray, V, trackedCorners, nextPts, lkStatus, lkErr,
            cv::Size(21, 21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01)
        );
    }

    // Оставляем только точки, которые LK смог уверенно проследить
    std::vector<cv::Point2f> prevGood, currGood;
    prevGood.reserve(nextPts.size());
    currGood.reserve(nextPts.size());
    for (size_t i = 0; i < nextPts.size(); i++) {
        if (lkStatus[i]) {
            prevGood.push_back(trackedCorners[i]);
            currGood.push_back(nextPts[i]);
        }
    }

    // ==========================================================
    // 2. Ищем гомографию фона через RANSAC.
    //    H переводит точки предыдущего кадра в текущий: p_curr ≈ H * p_prev.
    //    Точки, которые не подчиняются найденной H (outliers по мнению
    //    RANSAC) - это потенциально движущийся объект, а не фон/камера.
    // ==========================================================
    std::vector<uchar> ransacMask;
    cv::Mat Hnew;
    if (prevGood.size() >= 4) {
        Hnew = cv::findHomography(prevGood, currGood, cv::RANSAC,
                                   ransacReprojThreshold, ransacMask);
    }

    std::vector<cv::Point2f> backgroundPts, objectPts;
    std::vector<cv::Point2f> survivedCorners; // точки, которые понесём в следующий кадр
    std::vector<uchar> survivedStatus;

    if (!Hnew.empty()) {
        H = Hnew;
        for (size_t i = 0; i < currGood.size(); i++) {
            bool isBackground = ransacMask[i] != 0;
            if (isBackground) backgroundPts.push_back(currGood[i]);
            else               objectPts.push_back(currGood[i]);

            survivedCorners.push_back(currGood[i]);
            survivedStatus.push_back(isBackground ? 1 : 0);
        }
    } else {
        // Точек мало / гомография не найдена (например, резкий рывок камеры,
        // либо кадр почти целиком занят объектом) - считаем всё фоном,
        // H оставляем от предыдущего успешного кадра
        backgroundPts = currGood;
        survivedCorners = currGood;
        survivedStatus.assign(currGood.size(), 1);
    }

    // ==========================================================
    // 3. Компенсация движения камеры.
    //    "Перематываем" предыдущий кадр через H в систему координат
    //    текущего кадра и вычитаем из текущего. В идеале фон (который
    //    и описывает H) становится чёрным, а объект, движущийся
    //    независимо от камеры, остаётся цветным/ярким пятном.
    // ==========================================================
    cv::Mat diffFrame = cv::Mat::zeros(frame.size(), frame.type());
    if (!H.empty() && !prevFrame.empty()) {
        cv::Mat warpedPrev;
        cv::warpPerspective(prevFrame, warpedPrev, H, frame.size());
        cv::absdiff(frame, warpedPrev, diffFrame);
    }

    // ==========================================================
    // 4. Визуализация: зелёные точки - фон, красные - объект
    // ==========================================================
    cv::Mat vis = drawVisualization(frame, backgroundPts, objectPts);

    emit frameProcessed(vis, diffFrame);

    // ==========================================================
    // 5. Готовим состояние для следующего кадра
    // ==========================================================
    trackedCorners = survivedCorners;
    pointStatus = survivedStatus;

    // Пере-детект точек: либо раз в 30 кадров (как и раньше), либо когда
    // точек осталось слишком мало для устойчивой гомографии
    if (N % 30 == 0 || (int)trackedCorners.size() < minPointsToRedetect) {
        trackedCorners = detectCorners(frame);
        pointStatus.assign(trackedCorners.size(), 1);
    }

    prevGray = V.clone();
    prevFrame = frame.clone();

    N++;
}

cv::Mat Tracker::drawVisualization(const cv::Mat& frame,
                                    const std::vector<cv::Point2f>& backgroundPts,
                                    const std::vector<cv::Point2f>& objectPts) const
{
    cv::Mat vis = frame.clone();

    for (const auto& p : backgroundPts)
        cv::circle(vis, p, 4, cv::Scalar(0, 255, 0), -1); // зелёный - фон

    for (const auto& p : objectPts)
        cv::circle(vis, p, 4, cv::Scalar(0, 0, 255), -1); // красный - объект

    return vis;
}

std::vector<cv::Point2f> Tracker::detectCorners(const cv::Mat& frame)
{

    cv::Mat frameHSV;
    // Получает кадр в HSV (H - цвет, S - насыщенность, V - яркость)
    cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV); 
    

    // Берем канал V из HSV кадра
    std::vector<cv::Mat> channels;
    cv::split(frameHSV, channels);
    cv::Mat V = channels[2];  

    // ==========================================================
    // Разбиваем кадр на сетку. Если точка сетки является углом 
    // объекта на кадре то добавляем ее в отслеживаемые точки.
    // ==========================================================
    

    // ==========================================================
    // Сперва необходимо получить структурный
    // тензор для каждого пикселя сетки. 
    // Где Da - градиент изменения яркости пиксели по оси a
    // 
    // M = | Σ(Dx^2)   Σ(Dx*Dy) |
    //     | Σ(Dx*Dy)  Σ(Dy^2)  |
    //
    // ==========================================================

    // ==========================================================
    // Строим  3 карты (Dx^2 Dy^2 Dxy) изменения яркости пикселей 
    // ==========================================================
    int widthMAP = (V.size().width - step - 1) / step;
    int heightMAP = (V.size().height -step - 1) / step;
    int sizeMAP = widthMAP * heightMAP;
    std::vector<float> DX2(sizeMAP);
    std::vector<float> DY2(sizeMAP);
    std::vector<float> DXY(sizeMAP);

    int px = 0;
    float dx;
    float dy;     
    for(int y = step; y < V.size().height - step; y+=step){
        const uchar* rowUp   = V.ptr<uchar>(y - step);
        const uchar* rowCur  = V.ptr<uchar>(y);
        const uchar* rowDown = V.ptr<uchar>(y + step);

        for(int x = step; x < V.size().width - step; x+=step)
        {
            float dx =  (rowCur[x + step] - rowCur[x - step]) / 2.0f;
            float dy =  (rowDown[x] - rowUp[x]) / 2.0f;

            DX2[px]=dx*dx;
            DY2[px]=dy*dy;
            DXY[px]=dx*dy;

            px++;
        }
    } 
    
    // ==========================================================
    // Составление интегральных изображений 
    // (для оптимизированного расчета сумм )
    // ==========================================================
    std::vector<float> integralDX2 = buildIntegralImage(DX2, widthMAP, heightMAP);
    std::vector<float> integralDY2 = buildIntegralImage(DY2, widthMAP, heightMAP);
    std::vector<float> integralDXY = buildIntegralImage(DXY, widthMAP, heightMAP);

       
    // ==========================================================
    // Усреднение окном w * w
    // (получение сумм по типу Σ(Dx^2))
    // ==========================================================
    int halfWindow = windowSize /2;

    std::vector<float> DX2_blurred(sizeMAP, 0.0f);
    std::vector<float> DY2_blurred(sizeMAP, 0.0f);
    std::vector<float> DXY_blurred(sizeMAP, 0.0f);  

    int w = widthMAP + 1;
    for (int y = halfWindow; y < heightMAP - halfWindow; y++) {
        for (int x = halfWindow; x < widthMAP - halfWindow; x++) {
            int x1 = x - halfWindow;
            int y1 = y - halfWindow;
            int x2 = x + halfWindow + 1;
            int y2 = y + halfWindow + 1;
            
            float sumDX2 = integralDX2[y2 * w + x2] 
                         - integralDX2[y1 * w + x2] 
                         - integralDX2[y2 * w + x1] 
                         + integralDX2[y1 * w + x1];
            
            float sumDY2 = integralDY2[y2 * w + x2] 
                         - integralDY2[y1 * w + x2] 
                         - integralDY2[y2 * w + x1] 
                         + integralDY2[y1 * w + x1];
            
            float sumDXY = integralDXY[y2 * w + x2] 
                         - integralDXY[y1 * w + x2] 
                         - integralDXY[y2 * w + x1] 
                         + integralDXY[y1 * w + x1];
            
            int index = y * widthMAP + x;
            DX2_blurred[index] = sumDX2 / (windowSize*windowSize);
            DY2_blurred[index] = sumDY2 / (windowSize*windowSize);
            DXY_blurred[index] = sumDXY / (windowSize*windowSize);
        }
    }

    // ==========================================================
    // Получение отклика Харриса
    // ==========================================================
    // Для тензорной матрицы 
    //
    // M = | Σ(Dx^2)   Σ(Dx*Dy) |
    //     | Σ(Dx*Dy)  Σ(Dy^2)  |
    //
    // собственные значения:
    // λ = ((a+c) ± sqrt((a-c)^2 + 4b^2)) / 2

    std::vector<float> shiTomasi(sizeMAP, 0.0f);
    for (int i = 0; i < sizeMAP; i++) {
        float a = DX2_blurred[i];  // Ix²
        float b = DXY_blurred[i];  // Ix*Iy
        float c = DY2_blurred[i];  // Iy²

        float trace = a + c;
        float det = a * c - b * b;

        // Вычисляем собственные значения
        float discriminant = sqrt((a - c) * (a - c) + 4 * b * b);
        float lambda1 = (trace + discriminant) / 2.0f;
        float lambda2 = (trace - discriminant) / 2.0f;

        // Берем минимальное собственное значение (критерий Shi-Tomasi)
        shiTomasi[i] = std::min(lambda1, lambda2);
    }

    // 8. Находим углы 
    float maxResponse = *std::max_element(shiTomasi.begin(), shiTomasi.end());
    if (maxResponse <= 0.0f)
        return {};
    float thresh = shiTomasiThreshold * maxResponse; 

    std::vector<cv::Point2f> corners;
    for (int y = halfWindow; y < heightMAP - halfWindow; y++) {
        for (int x = halfWindow; x < widthMAP - halfWindow; x++) {
            int idx = y * widthMAP + x;
            float val = shiTomasi[idx];
            if (val <= thresh) continue;

            bool isLocalMax = true;
            for (int dy = -nmsRadius; dy <= nmsRadius && isLocalMax; dy++) {
                for (int dx = -nmsRadius; dx <= nmsRadius; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (nx < 0 || ny < 0 || nx >= widthMAP || ny >= heightMAP) continue;
                    if (shiTomasi[ny * widthMAP + nx] > val) {
                        isLocalMax = false;
                        break;
                    }
                }
            }

            if (isLocalMax) {
                corners.emplace_back(
                    static_cast<float>(x * step + step / 2),
                    static_cast<float>(y * step + step / 2)
                );
            }
        }
    }
    

    return corners;
}