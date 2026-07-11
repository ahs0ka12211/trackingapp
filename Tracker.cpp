#include "Tracker.h"

// Определения static constexpr членов
constexpr int Tracker::kBaseStep;
constexpr int Tracker::kBaseWindowSize;
constexpr int Tracker::kBaseNmsRadius;
constexpr int Tracker::kBaseMinPointsToRedetect;

Tracker::Tracker()
{
    N = 0;
    setGridStep(kBaseStep);

    shiTomasiThreshold = 0.048f;
    ransacReprojThreshold = 3.0f;
    centralZoneRatio = 0.68f;
    classificationZoneRatio = 0.88f;
    warpBorderErodePx = 12;

    qRegisterMetaType<cv::Mat>("cv::Mat");
}

void Tracker::setGridStep(int step_)
{
    if (step_ < 1) step_ = 1;
    step = step_;
    recalcStepDependentParams();
}

void Tracker::recalcStepDependentParams()
{
    float scale = static_cast<float>(kBaseStep) / static_cast<float>(step);

    int newWindow = static_cast<int>(std::round(kBaseWindowSize * scale));
    if (newWindow < 3) newWindow = 3;
    if (newWindow % 2 == 0) newWindow += 1;
    windowSize = newWindow;

    int newNms = static_cast<int>(std::round(kBaseNmsRadius * scale));
    if (newNms < 1) newNms = 1;
    nmsRadius = newNms;

    float densityRatio = scale * scale;
    int newMinPoints = static_cast<int>(std::round(kBaseMinPointsToRedetect * densityRatio));
    if (newMinPoints < 12) newMinPoints = 12;
    minPointsToRedetect = newMinPoints;

    qDebug() << "[Tracker] step =" << step
             << " -> windowSize =" << windowSize
             << ", nmsRadius =" << nmsRadius
             << ", minPointsToRedetect =" << minPointsToRedetect;
}

void Tracker::setDetectionParams(int diffThreshold_, int minObjectArea_,
                                  int openKernelSize_, int closeKernelSize_)
{
    diffThreshold = diffThreshold_;
    minObjectArea = minObjectArea_;
    openKernelSize = openKernelSize_;
    closeKernelSize = closeKernelSize_;
    lastKnownArea = 0.0;
}

// Построение интегрального изображения
std::vector<float> Tracker::buildIntegralImage(const std::vector<float>& data, int width, int height) {
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

    if (N == 0) {
        trackedCorners = detectCorners(V);
        pointStatus.assign(trackedCorners.size(), 1);
        anchorCorners = trackedCorners;
        framesSinceAnchor = 0;
        prevGray = V.clone();
        prevFrame = frame.clone();
        H = cv::Mat::eye(3, 3, CV_64F);
        N++;
        return;
    }

    // 1. Optical Flow
    std::vector<cv::Point2f> nextPts;
    std::vector<uchar> lkStatus;
    std::vector<float> lkErr;

    if (!trackedCorners.empty()) {
        int lkFlags = 0;

        if (!H.empty()) {
            cv::perspectiveTransform(trackedCorners, nextPts, H);
            lkFlags = cv::OPTFLOW_USE_INITIAL_FLOW;
        }

        float cameraMotionEst = cv::norm(H - cv::Mat::eye(3,3,CV_64F), cv::NORM_L2);
        int maxLevel = (cameraMotionEst > 12.0f) ? 5 : 4;

        cv::calcOpticalFlowPyrLK(
            prevGray, V, trackedCorners, nextPts, lkStatus, lkErr,
            cv::Size(21, 21), maxLevel,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            lkFlags
        );
    }

    std::vector<cv::Point2f> prevGood, currGood, anchorGood;
    for (size_t i = 0; i < nextPts.size(); i++) {
        if (lkStatus[i]) {
            prevGood.push_back(trackedCorners[i]);
            currGood.push_back(nextPts[i]);
            anchorGood.push_back(anchorCorners[i]);
        }
    }

    // 1.5. Длинная база (anchor)
    framesSinceAnchor++;

    cv::Mat Hanchor;
    bool haveAnchorH = false;
    if (framesSinceAnchor >= minAnchorBaselineFrames && anchorGood.size() >= 4) {
        std::vector<uchar> anchorMask;
        Hanchor = cv::findHomography(anchorGood, currGood, cv::RANSAC,
                                      anchorBaseReprojThreshold, anchorMask);
        haveAnchorH = !Hanchor.empty();
    }

    std::vector<cv::Point2f> warpedAnchor;
    if (haveAnchorH)
        cv::perspectiveTransform(anchorGood, warpedAnchor, Hanchor);

    float anchorThresh = anchorBaseReprojThreshold *
                          (1.0f + 0.25f * std::sqrt(static_cast<float>(framesSinceAnchor)));

    // 2. Гомография фона (улучшенная)
    std::vector<cv::Point2f> prevCentral, currCentral;
    for (size_t i = 0; i < prevGood.size(); i++) {
        if (isInCentralZone(currGood[i], frame.size(), centralZoneRatio)) {
            prevCentral.push_back(prevGood[i]);
            currCentral.push_back(currGood[i]);
        }
    }

    // Очистка от области объекта
    if (hasLastObject && !lastObjectBBox.empty()) {
        cv::Rect forbidden = lastObjectBBox;
        forbidden.x -= 50;
        forbidden.y -= 50;
        forbidden.width += 100;
        forbidden.height += 100;

        std::vector<cv::Point2f> pc, cc;
        for (size_t i = 0; i < prevCentral.size(); ++i) {
            if (!forbidden.contains(currCentral[i])) {
                pc.push_back(prevCentral[i]);
                cc.push_back(currCentral[i]);
            }
        }
        if (pc.size() >= 6) {
            prevCentral = std::move(pc);
            currCentral = std::move(cc);
        }
    }

    cv::Mat Hnew;
    float cameraMotion = 0.0f;
    if (prevCentral.size() >= 6) {
        std::vector<uchar> centralMask;
        int iterations = 2500;
        float reproj = ransacReprojThreshold * (1.0f + std::min(cameraMotion, 12.0f) * 0.12f);
        
        Hnew = cv::findHomography(prevCentral, currCentral, cv::RANSAC,
                                  reproj, centralMask, iterations);
    } else if (prevGood.size() >= 8) {
        std::vector<uchar> fallbackMask;
        Hnew = cv::findHomography(prevGood, currGood, cv::RANSAC,
                                ransacReprojThreshold * 1.4f, fallbackMask);
        qDebug() << "[Tracker] мало точек в центре, H считается по всему кадру";
    }

    std::vector<cv::Point2f> backgroundPts, objectPts;
    std::vector<cv::Point2f> survivedCorners;
    std::vector<uchar> survivedStatus;
    std::vector<cv::Point2f> survivedAnchor;

    if (!Hnew.empty()) {
        H = Hnew;
        cameraMotion = cv::norm(H - cv::Mat::eye(3,3,CV_64F), cv::NORM_L2);

        if (hasLastObject) {
            lastObjectBBox = warpBBox(lastObjectBBox, H, frame.size());
        }

        std::vector<cv::Point2f> warpedPrev;
        cv::perspectiveTransform(prevGood, warpedPrev, H);

        float dynamicReprojThreshold = ransacReprojThreshold *
            (1.0f + std::min(cameraMotion, 15.0f) * 0.12f);

        float currentRadialThresh = radialMotionThreshold;
        if (cameraMotion > 10.0f) currentRadialThresh = 0.48f;

        for (size_t i = 0; i < currGood.size(); i++) {
            float err = static_cast<float>(cv::norm(currGood[i] - warpedPrev[i]));
            bool isBackground = err <= dynamicReprojThreshold;

            // ====================== РАДИАЛЬНЫЙ БУСТ ======================
            if (isBackground && useRadialBoost && haveAnchorH && cameraMotion < radialBoostMaxCameraMotion) {
                cv::Point2f motion = currGood[i] - warpedPrev[i];
                cv::Point2f center(frame.cols/2.0f, frame.rows/2.0f);
                cv::Point2f toCenter = center - currGood[i];

                float motionLen = cv::norm(motion);
                float distToCenter = cv::norm(toCenter);

                if (motionLen > minRadialMotionLen && distToCenter > 30.0f) {
                    float radialness = std::abs(motion.dot(toCenter)) / 
                                      (motionLen * distToCenter + 1e-6f);

                    float errLong = haveAnchorH ? 
                        static_cast<float>(cv::norm(currGood[i] - warpedAnchor[i])) : 0;

                    if (radialness > currentRadialThresh && errLong > anchorThresh * 0.55f) {
                        isBackground = false;
                    }
                }
            }
            // ============================================================

            // Длинная база
            if (isBackground && haveAnchorH) {
                float errLong = static_cast<float>(cv::norm(currGood[i] - warpedAnchor[i]));
                if (errLong > anchorThresh)
                    isBackground = false;
            }

            // Защита от параллакса
            if (isBackground && hasLastObject && isNearLastObject(currGood[i], 45.0f)) {
                float errLong = haveAnchorH ? 
                    static_cast<float>(cv::norm(currGood[i] - warpedAnchor[i])) : 0;
                if (errLong > anchorThresh * 0.65f)
                    isBackground = false;
            }

            // Краевые точки — фон
            if (!isInCentralZone(currGood[i], frame.size(), classificationZoneRatio))
                isBackground = true;

            if (isBackground) backgroundPts.push_back(currGood[i]);
            else               objectPts.push_back(currGood[i]);

            survivedCorners.push_back(currGood[i]);
            survivedStatus.push_back(isBackground ? 1 : 0);
            survivedAnchor.push_back(anchorGood[i]);
        }
    } else {
        backgroundPts = currGood;
        survivedCorners = currGood;
        survivedStatus.assign(currGood.size(), 1);
        survivedAnchor = anchorGood;
    }

    // 3. Компенсация + diff
    cv::Mat diffFrame = cv::Mat::zeros(frame.size(), frame.type());
    cv::Mat shadowMask;
    if (!H.empty() && !prevFrame.empty()) {
        cv::Mat warpedPrev;
        cv::warpPerspective(prevFrame, warpedPrev, H, frame.size());
        cv::absdiff(frame, warpedPrev, diffFrame);

        cv::Mat validMask(prevFrame.size(), CV_8UC1, cv::Scalar(255));
        cv::warpPerspective(validMask, validMask, H, frame.size());
        cv::erode(validMask, validMask,
                  cv::getStructuringElement(cv::MORPH_ELLIPSE,
                      cv::Size(warpBorderErodePx, warpBorderErodePx)));
        diffFrame.setTo(cv::Scalar::all(0), validMask == 0);

        shadowMask = computeShadowMask(frame, warpedPrev);
        if (!shadowMask.empty()) {
            shadowMask.setTo(0, validMask == 0);
            diffFrame.setTo(cv::Scalar::all(0), shadowMask);
        }
    }

    // Переклассификация по тени
    if (!shadowMask.empty()) {
        for (size_t i = 0; i < survivedCorners.size(); i++) {
            if (survivedStatus[i] == 0) {
                const cv::Point2f& pt = survivedCorners[i];
                int x = cvRound(pt.x);
                int y = cvRound(pt.y);
                if (x >= 0 && y >= 0 && x < shadowMask.cols && y < shadowMask.rows &&
                    shadowMask.at<uchar>(y, x) > 0) {
                    survivedStatus[i] = 1;
                }
            }
        }

        backgroundPts.clear();
        objectPts.clear();
        for (size_t i = 0; i < survivedCorners.size(); i++) {
            if (survivedStatus[i]) backgroundPts.push_back(survivedCorners[i]);
            else                   objectPts.push_back(survivedCorners[i]);
        }
    }

    // Усиление diff feature points объекта
    if (!objectPts.empty()) {
        cv::Mat pointsMask = cv::Mat::zeros(diffFrame.size(), CV_8U);
        for (const auto& pt : objectPts) {
            int x = cvRound(pt.x);
            int y = cvRound(pt.y);
            if (x >= 0 && x < pointsMask.cols && y >= 0 && y < pointsMask.rows) {
                cv::circle(pointsMask, cv::Point(x, y), 7, cv::Scalar(255), -1);
            }
        }
        cv::dilate(pointsMask, pointsMask, cv::Mat(), cv::Point(-1,-1), 3);
        
        cv::Mat grayDiff;
        if (diffFrame.channels() == 3)
            cv::cvtColor(diffFrame, grayDiff, cv::COLOR_BGR2GRAY);
        else
            grayDiff = diffFrame;
        
        cv::bitwise_or(grayDiff, pointsMask, grayDiff);
        if (diffFrame.channels() == 3) {
            cv::cvtColor(grayDiff, diffFrame, cv::COLOR_GRAY2BGR);
        } else {
            diffFrame = grayDiff;
        }
    }

    // 4. Детекция объекта
    cv::Rect objectBBox = detectMovingObjectBBox(diffFrame, objectPts, minObjectArea);

    cv::Mat vis = drawVisualization(frame, backgroundPts, objectPts);

    if (objectBBox.area() > 0) {
        cv::rectangle(vis, objectBBox, cv::Scalar(0, 255, 255), 2);
    }

    emit frameProcessed(vis, diffFrame);

    // 5. Обновление состояния
    trackedCorners = survivedCorners;
    pointStatus = survivedStatus;
    anchorCorners = survivedAnchor;

    if (N % 25 == 0 || (int)trackedCorners.size() < minPointsToRedetect) {
        trackedCorners = detectCorners(V);
        pointStatus.assign(trackedCorners.size(), 1);
        anchorCorners = trackedCorners;
        framesSinceAnchor = 0;
    }
    else if (framesSinceAnchor >= (cameraMotion > 9.0f ? 6 : anchorRefreshInterval)) {
        anchorCorners = trackedCorners;
        framesSinceAnchor = 0;
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
        cv::circle(vis, p, 4, cv::Scalar(0, 255, 0), -1);

    for (const auto& p : objectPts)
        cv::circle(vis, p, 4, cv::Scalar(0, 0, 255), -1);

    return vis;
}

std::vector<cv::Point2f> Tracker::detectCorners(const cv::Mat& V)
{
    int widthMAP = (V.size().width - step - 1) / step;
    int heightMAP = (V.size().height - step - 1) / step;
    int sizeMAP = widthMAP * heightMAP;
    std::vector<float> DX2(sizeMAP);
    std::vector<float> DY2(sizeMAP);
    std::vector<float> DXY(sizeMAP);

    int px = 0;
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
    
    std::vector<float> integralDX2 = buildIntegralImage(DX2, widthMAP, heightMAP);
    std::vector<float> integralDY2 = buildIntegralImage(DY2, widthMAP, heightMAP);
    std::vector<float> integralDXY = buildIntegralImage(DXY, widthMAP, heightMAP);

    int halfWindow = windowSize / 2;

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
            
            float sumDX2 = integralDX2[y2 * w + x2] - integralDX2[y1 * w + x2] 
                         - integralDX2[y2 * w + x1] + integralDX2[y1 * w + x1];
            
            float sumDY2 = integralDY2[y2 * w + x2] - integralDY2[y1 * w + x2] 
                         - integralDY2[y2 * w + x1] + integralDY2[y1 * w + x1];
            
            float sumDXY = integralDXY[y2 * w + x2] - integralDXY[y1 * w + x2] 
                         - integralDXY[y2 * w + x1] + integralDXY[y1 * w + x1];
            
            int index = y * widthMAP + x;
            DX2_blurred[index] = sumDX2 / (windowSize*windowSize);
            DY2_blurred[index] = sumDY2 / (windowSize*windowSize);
            DXY_blurred[index] = sumDXY / (windowSize*windowSize);
        }
    }

    std::vector<float> shiTomasi(sizeMAP, 0.0f);
    for (int i = 0; i < sizeMAP; i++) {
        float a = DX2_blurred[i];
        float b = DXY_blurred[i];
        float c = DY2_blurred[i];

        float trace = a + c;
        float det = a * c - b * b;
        float discriminant = sqrt((a - c) * (a - c) + 4 * b * b);
        float lambda1 = (trace + discriminant) / 2.0f;
        float lambda2 = (trace - discriminant) / 2.0f;

        shiTomasi[i] = std::min(lambda1, lambda2);
    }

    float maxResponse = *std::max_element(shiTomasi.begin(), shiTomasi.end());
    if (maxResponse <= 0.0f) return {};
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

// ====================== detectMovingObjectBBox ======================
cv::Rect Tracker::detectMovingObjectBBox(const cv::Mat& diffFrame, 
                                         const std::vector<cv::Point2f>& objectFeaturePts,
                                         int minArea)
{
    if (diffFrame.empty()) return cv::Rect();

    cv::Mat gray;
    if (diffFrame.channels() == 3)
        cv::cvtColor(diffFrame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = diffFrame;

    cv::Mat mask;
    cv::threshold(gray, mask, diffThreshold, 255, cv::THRESH_BINARY);

    // Motion Evidence
    if (motionEvidence.empty() || motionEvidence.size() != gray.size()) {
        motionEvidence = cv::Mat::zeros(gray.size(), CV_32F);
    }

    cv::Mat weakMask;
    cv::threshold(gray, weakMask, diffThreshold / 2, 1.0, cv::THRESH_BINARY);
    weakMask.convertTo(weakMask, CV_32F);
    cv::accumulateWeighted(weakMask, motionEvidence, motionEvidenceAlpha);

    cv::Mat evidenceMask;
    cv::threshold(motionEvidence, evidenceMask, motionEvidenceThreshold, 255, cv::THRESH_BINARY);
    evidenceMask.convertTo(evidenceMask, CV_8U);

    cv::bitwise_or(mask, evidenceMask, mask);

    // Морфология
    if (openKernelSize > 0) {
        cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                  cv::Size(openKernelSize, openKernelSize));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelOpen);
    }

    if (closeKernelSize > 0) {
        cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                   cv::Size(closeKernelSize, closeKernelSize));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelClose);
    }

    // Feature points boost
    if (!objectFeaturePts.empty()) {
        cv::Mat featMask = cv::Mat::zeros(mask.size(), CV_8U);
        for (const auto& pt : objectFeaturePts) {
            int x = cvRound(pt.x);
            int y = cvRound(pt.y);
            if (x >= 0 && x < featMask.cols && y >= 0 && y < featMask.rows)
                cv::circle(featMask, cv::Point(x,y), 8, cv::Scalar(255), -1);
        }
        cv::dilate(featMask, featMask, cv::Mat(), cv::Point(-1,-1), 3);
        cv::bitwise_or(mask, featMask, mask);
    }

    std::vector<std::vector<cv::Point>> contours;
    int borderX = static_cast<int>(mask.cols * 0.05);
    int borderY = static_cast<int>(mask.rows * 0.05);
    mask(cv::Rect(0, 0, mask.cols, borderY)).setTo(0);
    mask(cv::Rect(0, mask.rows - borderY, mask.cols, borderY)).setTo(0);
    mask(cv::Rect(0, 0, borderX, mask.rows)).setTo(0);
    mask(cv::Rect(mask.cols - borderX, 0, borderX, mask.rows)).setTo(0);

    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> candidates;
    std::vector<double> areas;
    std::vector<int> objectPointCounts;

    int adaptiveMinArea = minObjectArea;
    if (hasLastObject && lastKnownArea > 700) {
        adaptiveMinArea = std::max(100, static_cast<int>(lastKnownArea * 0.22));
    }

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < adaptiveMinArea) continue;

        cv::Rect r = cv::boundingRect(c);
        candidates.push_back(r);
        areas.push_back(area);

        int count = 0;
        for (const auto& pt : objectFeaturePts) {
            if (r.contains(cv::Point(cvRound(pt.x), cvRound(pt.y)))) count++;
        }
        objectPointCounts.push_back(count);
    }

    if (candidates.empty()) {
        if (hasLastObject) {
            missedFrames++;
            if (missedFrames > maxMissedFrames) {
                hasLastObject = false;
                missedFrames = 0;
                lastKnownArea = 0.0;
            }
        }
        return cv::Rect();
    }

    cv::Point2f frameCenter(gray.cols / 2.0f, gray.rows / 2.0f);
    int bestIdx = -1;

    if (hasLastObject) {
        cv::Point2f lastCenter(lastObjectBBox.x + lastObjectBBox.width / 2.0f,
                               lastObjectBBox.y + lastObjectBBox.height / 2.0f);
        float searchRadius = std::max(lastObjectBBox.width, lastObjectBBox.height) * 4.0f + 80.0f;

        double bestScore = -1e9;
        for (size_t i = 0; i < candidates.size(); i++) {
            cv::Point2f c(candidates[i].x + candidates[i].width / 2.0f,
                          candidates[i].y + candidates[i].height / 2.0f);
            double dist = cv::norm(c - lastCenter);
            if (dist > searchRadius) continue;

            double score = areas[i] * 0.55 + objectPointCounts[i] * 32.0 - dist * 0.7;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = static_cast<int>(i);
            }
        }
    } else {
        double bestScore = -1e9;
        for (size_t i = 0; i < candidates.size(); i++) {
            cv::Point2f c(candidates[i].x + candidates[i].width / 2.0f,
                          candidates[i].y + candidates[i].height / 2.0f);
            double distToCenter = cv::norm(c - frameCenter);
            double score = areas[i] * 0.5 + objectPointCounts[i] * 30.0 
                         - distToCenter * 0.4;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = static_cast<int>(i);
            }
        }
    }

    if (bestIdx >= 0) {
        lastObjectBBox = candidates[bestIdx];
        lastKnownArea = areas[bestIdx];
        hasLastObject = true;
        missedFrames = 0;
        return lastObjectBBox;
    }

    if (hasLastObject) {
        missedFrames++;
        if (missedFrames > maxMissedFrames) {
            hasLastObject = false;
            missedFrames = 0;
            lastKnownArea = 0.0;
        }
    }
    return cv::Rect();
}

// ====================== computeShadowMask ======================
cv::Mat Tracker::computeShadowMask(const cv::Mat& currFrame, const cv::Mat& bgFrame) const
{
    if (currFrame.empty() || bgFrame.empty() || currFrame.size() != bgFrame.size())
        return cv::Mat();

    cv::Mat currHSV, bgHSV;
    cv::cvtColor(currFrame, currHSV, cv::COLOR_BGR2HSV);
    cv::cvtColor(bgFrame, bgHSV, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> currCh, bgCh;
    cv::split(currHSV, currCh);
    cv::split(bgHSV, bgCh);

    const cv::Mat& curH = currCh[0];
    const cv::Mat& curS = currCh[1];
    const cv::Mat& curV = currCh[2];
    const cv::Mat& bgH  = bgCh[0];
    const cv::Mat& bgS  = bgCh[1];
    const cv::Mat& bgV  = bgCh[2];

    cv::Mat diffS, diffH;
    cv::absdiff(curS, bgS, diffS);
    cv::absdiff(curH, bgH, diffH);
    cv::Mat diffHWrapped = cv::min(diffH, 180 - diffH);

    cv::Mat shadowMask = cv::Mat::zeros(currFrame.size(), CV_8U);

    for (int y = 0; y < shadowMask.rows; y++) {
        const uchar* rowCurV = curV.ptr<uchar>(y);
        const uchar* rowBgV  = bgV.ptr<uchar>(y);
        const uchar* rowDiffS = diffS.ptr<uchar>(y);
        const uchar* rowDiffH = diffHWrapped.ptr<uchar>(y);
        uchar* rowOut = shadowMask.ptr<uchar>(y);

        for (int x = 0; x < shadowMask.cols; x++) {
            float vCur = static_cast<float>(rowCurV[x]);
            float vBg  = static_cast<float>(rowBgV[x]) + 1.0f;
            float ratio = vCur / vBg;

            bool darkerButNotTooMuch = (ratio >= shadowVRatioMin) && (ratio <= shadowVRatioMax);
            bool hueStable = rowDiffH[x] <= shadowHueDiffMax;
            bool satStable = rowDiffS[x] <= shadowSatDiffMax;

            if (darkerButNotTooMuch && hueStable && satStable)
                rowOut[x] = 255;
        }
    }

    if (shadowOpenKernelSize > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                             cv::Size(shadowOpenKernelSize, shadowOpenKernelSize));
        cv::morphologyEx(shadowMask, shadowMask, cv::MORPH_OPEN, kernel);
    }

    return shadowMask;
}

bool Tracker::isInCentralZone(const cv::Point2f& p, const cv::Size& frameSize, float zoneRatio) const
{
    float marginX = frameSize.width  * (1.0f - zoneRatio) / 2.0f;
    float marginY = frameSize.height * (1.0f - zoneRatio) / 2.0f;

    return p.x >= marginX && p.x <= frameSize.width  - marginX &&
           p.y >= marginY && p.y <= frameSize.height - marginY;
}

bool Tracker::isNearLastObject(const cv::Point2f& p, float margin) const
{
    if (!hasLastObject) return false;
    cv::Rect expanded = lastObjectBBox;
    expanded.x -= static_cast<int>(margin);
    expanded.y -= static_cast<int>(margin);
    expanded.width += static_cast<int>(margin * 2);
    expanded.height += static_cast<int>(margin * 2);
    return expanded.contains(cv::Point(cvRound(p.x), cvRound(p.y)));
}

cv::Rect Tracker::warpBBox(const cv::Rect& bbox, const cv::Mat& H, const cv::Size& frameSize) const
{
    if (H.empty() || bbox.area() <= 0) return bbox;

    std::vector<cv::Point2f> corners = {
        cv::Point2f(static_cast<float>(bbox.x), static_cast<float>(bbox.y)),
        cv::Point2f(static_cast<float>(bbox.x + bbox.width), static_cast<float>(bbox.y)),
        cv::Point2f(static_cast<float>(bbox.x + bbox.width), static_cast<float>(bbox.y + bbox.height)),
        cv::Point2f(static_cast<float>(bbox.x), static_cast<float>(bbox.y + bbox.height))
    };

    std::vector<cv::Point2f> warped;
    cv::perspectiveTransform(corners, warped, H);

    cv::Rect result = cv::boundingRect(warped);

    float areaRatio = static_cast<float>(result.area()) / (bbox.area() + 1.0f);
    if (areaRatio < 0.25f || areaRatio > 4.5f) {
        return bbox;
    }

    cv::Rect sane(-frameSize.width, -frameSize.height,
                  frameSize.width * 3, frameSize.height * 3);
    cv::Rect clipped = result & sane;
    return clipped.area() > 0 ? clipped : bbox;
}  