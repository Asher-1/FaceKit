#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

#define M_PI  3.14159265358979323846
#define CLAMP(x, l, u)  ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
#define EPS  1e-5

#define CYAN CV_RGB(0, 255, 255)
#define BLUE CV_RGB(0, 0, 255)
#define GREEN CV_RGB(0, 255, 0)
#define RED CV_RGB(255, 0, 0)
#define PURPLE CV_RGB(139, 0, 255)

#define kFeaturePoints 14

struct Window
{
    int x, y, width,height;
    float angle, scale;
    float conf;
    long id;
    cv::Point points14[kFeaturePoints];

    Window(int x_, int y_, int w_, int h_, float a_, float s_, float c_, long id_, cv::Point p14_[kFeaturePoints])
        : x(x_), y(y_), width(w_),height(h_), angle(a_), scale(s_), conf(c_), id(id_)
    {
	    set_points(p14_);
    }
    
    //New window without points and ID
    Window(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
        : x(x_), y(y_), width(w_),height(h_), angle(a_), scale(s_), conf(c_), id(-1)
    {}

    void set_points(cv::Point p14_[])
    {
	    memcpy(points14,&(p14_[0]),kFeaturePoints*sizeof(cv::Point));
    }
};

cv::Point RotatePoint(float x, float y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    float rx = centerX + x * std::cos(theta) - y * std::sin(theta);
    float ry = centerY + x * std::sin(theta) + y * std::cos(theta);
    return cv::Point(rx, ry);
}

void DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int width = 2;
    cv::line(img, pointList[0], pointList[1], CYAN, width);
    cv::line(img, pointList[1], pointList[2], CYAN, width);
    cv::line(img, pointList[2], pointList[3], CYAN, width);
    cv::line(img, pointList[3], pointList[0], BLUE, width);
}

void DrawFace(cv::Mat img, Window face)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y1, centerX, centerY, face.angle));
    DrawLine(img, pointList);
    cv::putText(img, std::string("id:") + std::to_string(face.id),
		    cv::Point(x1, y1), 2, 1, cv::Scalar(255, 0, 0));
}



class PCN
{
public:
    PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
        std::string modelTrack, std::string netTrack);
    /// detection
    void SetMinFaceSize(int minFace);
    void SetDetectionThresh(float thresh1, float thresh2, float thresh3);
    void SetImagePyramidScaleFactor(float factor);
    std::vector<Window> Detect(cv::Mat img);
    /// tracking
    void SetTrackingPeriod(int period);
    void SetTrackingThresh(float thresh);
    std::vector<Window> DetectTrack(cv::Mat img);
    int GetTrackingFrame();
    static cv::Mat CropFace(cv::Mat img, Window face, int cropSize);
    static void DrawPoints(cv::Mat img, Window face); 
    static void DrawFace(cv::Mat img, Window face);
    static void DrawLine(cv::Mat img, std::vector<cv::Point> pointList);
    static cv::Point RotatePoint(float x, float y, float centerX, float centerY, float angle);
private:
    void* impl_;
};

#endif
