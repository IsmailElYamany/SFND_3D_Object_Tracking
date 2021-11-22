
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> matches;

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    { // outer kpt. loop

        cv::KeyPoint kpCurr = kptsCurr.at(it1->trainIdx);

        cv::Rect roi = boundingBox.roi;
        if (roi.contains(kpCurr.pt))
        {

            matches.push_back(*it1);
        }
    }

    double sum = 0;
    double mean = 0;
    double max = 0;
    double min = 1e9;
    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1)
    {
        sum += it1->distance;
        if (it1->distance < min)
            min = it1->distance;
        if (it1->distance > max)
            max = it1->distance;
    }
    if (matches.size() > 0)
    {
        mean = sum / (matches.size());
    }
    else
        return;

    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1)
    {
        if (it1->distance > 0.7 * mean && it1->distance < 1 * mean)
        {
            boundingBox.kptMatches.push_back(*it1);
        }
    }

    std::cout << "the mean is : " << mean << "  The max is: " << max << "  The min is: " << min << endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat &visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    std::vector<cv::KeyPoint> kptsCurrMat;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        kptsCurrMat.push_back(kpOuterCurr);
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    cv::Mat img = visImg.clone();
    cv::drawKeypoints(img, kptsCurrMat, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    //double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);

    // TODO: STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long index = floor(distRatios.size() / 2.0);
    double medianDistRatio = distRatios.size() % 2 == 0 ? (distRatios[index - 1] + distRatios[index]) / 2.0 : distRatios[index]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 0.1; // time between two measurements in seconds
    std::vector<double> lidarPrev;
    std::vector<double> lidarCurr;

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        minXPrev = minXPrev > it->x ? it->x : minXPrev;
        lidarPrev.push_back(it->x);
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        minXCurr = minXCurr > it->x ? it->x : minXCurr;
        lidarCurr.push_back(it->x);
    }

    std::sort(lidarPrev.begin(), lidarPrev.end());
    long index = floor(lidarPrev.size() / 2.0);
    double medianXPrev = lidarPrev.size() % 2 == 0 ? (lidarPrev[index - 1] + lidarPrev[index]) / 2.0 : lidarPrev[index]; // compute median dist. ratio to remove outlier influence

    std::sort(lidarCurr.begin(), lidarCurr.end());
    index = floor(lidarCurr.size() / 2.0);
    double medianXCurr = lidarCurr.size() % 2 == 0 ? (lidarCurr[index - 1] + lidarCurr[index]) / 2.0 : lidarCurr[index]; // compute median dist. ratio to remove outlier influence

    // compute TTC from both measurements
    // TTC = minXCurr * dT / (minXPrev - minXCurr);
    TTC = medianXCurr * dT / (medianXPrev - medianXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    std::map<pair<int, int>, int> counts;
    for (int i = 0; i < matches.size(); i++)
    {
        int idx1 = matches[i].trainIdx; //curr
        int idx2 = matches[i].queryIdx; //prev
        cv::Point2f currPoint = currFrame.keypoints[idx1].pt;
        cv::Point2f prevPoint = prevFrame.keypoints[idx2].pt;

        for (const BoundingBox &prevbbox : prevFrame.boundingBoxes)
        {
            for (const BoundingBox &currbbox : currFrame.boundingBoxes)
            {
                if (prevbbox.roi.contains(prevPoint) && currbbox.roi.contains(currPoint))
                {
                    const int prevbboxid = prevbbox.boxID;
                    const int currbboxid = currbbox.boxID;

                    // Check if we have encountered the same combination of boxes before
                    std::map<std::pair<int, int>, int>::iterator it =
                        counts.find(std::make_pair(prevbboxid, currbboxid));
                    if (it == counts.end())
                    {
                        // If we haven't, make a new entry to start counting
                        counts.insert(std::pair<std::pair<int, int>, int>(
                            std::make_pair(prevbboxid, currbboxid), 1));
                    }
                    else
                    {
                        const int count = it->second;
                        it->second = count + 1;
                    }
                }
            }
        }
    }

    // Collect all unique bounding box IDs from the previous frame
    set<int> prev_ids;
    for (const auto &it : counts)
    {
        prev_ids.insert(it.first.first);
    }

    // Now loop over all possible bounding box IDs from the previous frame,
    // see how many combinations of matches to the current frame we have
    // then choose the largest one
    for (const int box_id : prev_ids)
    {
        // Record the largest amount of matches
        int max_count = -1;

        // Stores the current bounding box ID with the largest matches
        int curr_box_id = -1;

        // Loop through all combinations
        for (const auto &it : counts)
        {
            // If this isn't the previous bounding box ID
            // we're concentrating on, skip
            if (it.first.first != box_id)
            {
                continue;
            }

            // If this prev., curr. pair has a count
            // that exceeds the max...
            if (it.second > max_count)
            {
                // Record it and the current bounding box
                // ID that it's connected to
                max_count = it.second;
                curr_box_id = it.first.second;
            }
        }

        // If there are no matches from the previous frame
        // to any boxes to the current frame, skip
        if (curr_box_id == -1)
        {
            continue;
        }
        bbBestMatches.insert(std::pair<int, int>(box_id, curr_box_id));
    }
}