#include "Tracker.h"

Tracker::Tracker()
{
    step = 5; N = 0;
    windowSize = 7;
    shiTomasiThreshold = 0.05f;          
    minPointsToRedetect = 30;                
    nmsRadius = 2;    
    ransacReprojThreshold = 3.0f;
    centralZoneRatio = 0.7f; // используем центральные 70% кадра для оценки гомографии
    classificationZoneRatio = 0.9f; // точки в приграничных 10% кадра всегда считаем фоном
    warpBorderErodePx = 9;          // на сколько пикселей сжимать маску валидности warp'а

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


void Tracker::getFrame(const cv::Mat frame){
 
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
    //    Оцениваем H только по точкам из центральной зоны кадра -
    //    на краях сильнее проявляется параллакс (объекты на разной
    //    глубине сдвигаются по-разному при повороте/зуме камеры,
    //    а гомография описывает движение только одной плоскости).
    //    Классификацию background/object делаем ПОСЛЕ, по всем точкам,
    //    через фактическую ошибку репроекции найденной H.
    // ==========================================================
    std::vector<cv::Point2f> prevCentral, currCentral;
    prevCentral.reserve(prevGood.size());
    currCentral.reserve(currGood.size());
    for (size_t i = 0; i < prevGood.size(); i++) {
        // Зону проверяем по текущему кадру - именно в его системе
        // координат мы потом будем классифицировать точки
        if (isInCentralZone(currGood[i], frame.size(), centralZoneRatio)) {
            prevCentral.push_back(prevGood[i]);
            currCentral.push_back(currGood[i]);
        }
    }

    cv::Mat Hnew;
    if (prevCentral.size() >= 4) {
        std::vector<uchar> centralMask; // используется только внутри RANSAC, нам не нужна
        Hnew = cv::findHomography(prevCentral, currCentral, cv::RANSAC,
                                ransacReprojThreshold, centralMask);
    } else if (prevGood.size() >= 4) {
        // Фоллбэк: если в центре осталось слишком мало точек
        // (например, объект целиком закрыл центр кадра) - берём все,
        // неидеальная H лучше, чем её отсутствие
        std::vector<uchar> fallbackMask;
        Hnew = cv::findHomography(prevGood, currGood, cv::RANSAC,
                                ransacReprojThreshold, fallbackMask);
        qDebug() << "[Tracker] мало точек в центре, H считается по всему кадру";
    }

    std::vector<cv::Point2f> backgroundPts, objectPts;
    std::vector<cv::Point2f> survivedCorners; // точки, которые понесём в следующий кадр
    std::vector<uchar> survivedStatus;

    if (!Hnew.empty()) {
        H = Hnew;

        // Классифицируем ВСЕ точки (включая краевые) по реальной ошибке
        // репроекции через H, а не по RANSAC-маске (та считалась только
        // для центральной подвыборки и не соответствует размеру currGood)
        std::vector<cv::Point2f> warpedPrev;
        cv::perspectiveTransform(prevGood, warpedPrev, H);

        for (size_t i = 0; i < currGood.size(); i++) {
            float err = static_cast<float>(cv::norm(currGood[i] - warpedPrev[i]));
            bool isBackground = err <= ransacReprojThreshold;

            // У самой рамки кадра reprojection error естественно выше даже для
            // настоящего фона (дисторсия объектива, параллакс, экстраполяция H
            // за пределы зоны, по которой она считалась) - жёсткий порог там
            // ошибочно принимает фон за объект. Поэтому такие точки всегда
            // считаем фоном, не давая им "стать" ложным объектом.
            if (!isInCentralZone(currGood[i], frame.size(), classificationZoneRatio))
                isBackground = true;

            if (isBackground) backgroundPts.push_back(currGood[i]);
            else               objectPts.push_back(currGood[i]);

            survivedCorners.push_back(currGood[i]);
            survivedStatus.push_back(isBackground ? 1 : 0);
        }
    } else {
        // Гомография не найдена совсем - считаем всё фоном,
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

        // При панораме/зуме у warpedPrev по краям появляются зоны, для которых
        // в prevFrame нет исходных данных - warpPerspective заливает их чёрным
        // (BORDER_CONSTANT). absdiff с реальным содержимым текущего кадра в этих
        // местах даёт огромный ложный diff, который detectMovingObjectBBox
        // потом принимает за объект. Строим маску валидности тем же H и
        // обнуляем diff там, где реальных данных нет.
        cv::Mat validMask(prevFrame.size(), CV_8UC1, cv::Scalar(255));
        cv::warpPerspective(validMask, validMask, H, frame.size());
        cv::erode(validMask, validMask,
                  cv::getStructuringElement(cv::MORPH_ELLIPSE,
                      cv::Size(warpBorderErodePx, warpBorderErodePx)));
        diffFrame.setTo(cv::Scalar::all(0), validMask == 0);
    }

    // ==========================================================
    // 4. Визуализация: зелёные точки - фон, красные - объект
    // ==========================================================
    cv::Rect objectBBox = detectMovingObjectBBox(diffFrame, minObjectArea);

    cv::Mat vis = drawVisualization(frame, backgroundPts, objectPts);

    if (objectBBox.area() > 0) {
        cv::rectangle(vis, objectBBox, cv::Scalar(0, 255, 255), 2); // жёлтый прямоугольник
    }

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

cv::Rect Tracker::detectMovingObjectBBox(const cv::Mat& diffFrame, int minArea)
{
    if (diffFrame.empty()) return cv::Rect();

    cv::Mat gray;
    if (diffFrame.channels() == 3)
        cv::cvtColor(diffFrame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = diffFrame;

    // Отсекаем слабый шум компенсации (мелкие несовпадения из-за неидеальной H)
    cv::Mat mask;
    cv::threshold(gray, mask, diffThreshold, 255, cv::THRESH_BINARY);

    // Открытие — убирает одиночные шумные пиксели
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelOpen);

    // Закрытие — склеивает разорванный силуэт объекта в одно пятно
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelClose);

    std::vector<std::vector<cv::Point>> contours;

    int borderX = static_cast<int>(mask.cols * 0.05);
    int borderY = static_cast<int>(mask.rows * 0.05);
    mask(cv::Rect(0, 0, mask.cols, borderY)).setTo(0);
    mask(cv::Rect(0, mask.rows - borderY, mask.cols, borderY)).setTo(0);
    mask(cv::Rect(0, 0, borderX, mask.rows)).setTo(0);
    mask(cv::Rect(mask.cols - borderX, 0, borderX, mask.rows)).setTo(0);

    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect bestRect;
    double bestArea = 0.0;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea) continue;
        if (area > bestArea) {
            bestArea = area;
            bestRect = cv::boundingRect(c);
        }
    }

    return bestRect; // (0,0,0,0), если ничего не нашли
}

bool Tracker::isInCentralZone(const cv::Point2f& p, const cv::Size& frameSize, float zoneRatio) const
{
    float marginX = frameSize.width  * (1.0f - zoneRatio) / 2.0f;
    float marginY = frameSize.height * (1.0f - zoneRatio) / 2.0f;

    return p.x >= marginX && p.x <= frameSize.width  - marginX &&
           p.y >= marginY && p.y <= frameSize.height - marginY;
}