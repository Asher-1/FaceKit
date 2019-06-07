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

    void set_points(cv::Point p14_[]) {
	    memcpy(points14,&(p14_[0]),kFeaturePoints*sizeof(cv::Point));
    }
};

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
    void LoadModel_(std::string modelDetect, std::string net1, std::string net2, std::string net3,
                   std::string modelTrack, std::string netTrack);
    cv::Mat ResizeImg_(cv::Mat img, float scale);
    static bool CompareWin_(const Window &w1, const Window &w2);
    bool Legal_(int x, int y, cv::Mat img);
    bool Inside_(int x, int y, Window rect);
    int SmoothAngle_(int a, int b);
    std::vector<Window> SmoothWindowWithId_(std::vector<Window> winList);
    float IoU_(Window &w1, Window &w2);
    std::vector<Window> NMS_(std::vector<Window> &winList, bool local, float threshold);
    std::vector<Window> DeleteFP_(std::vector<Window> &winList);
    cv::Mat PreProcessImg_(cv::Mat img);
    cv::Mat PreProcessImg_(cv::Mat img,  int dim);
    void SetInput_(cv::Mat input, caffe::shared_ptr<caffe::Net<float> > &net);
    void SetInput_(std::vector<cv::Mat> &input, caffe::shared_ptr<caffe::Net<float> > &net);
    cv::Mat PadImg_(cv::Mat img);
    std::vector<Window> TransWindow_(cv::Mat img, cv::Mat imgPad, std::vector<Window> &winList);
    std::vector<Window> Stage1_(cv::Mat img, cv::Mat imgPad, caffe::shared_ptr<caffe::Net<float> > &net, float thres);
    std::vector<Window> Stage2_(cv::Mat img, cv::Mat img180,
                                caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList);
    std::vector<Window> Stage3_(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90,
                                caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList);
    std::vector<Window> Detect_(cv::Mat img, cv::Mat imgPad);
    std::vector<Window> Track_(cv::Mat img, caffe::shared_ptr<caffe::Net<float> > &net,
                               float thres, int dim, std::vector<Window> &winList);

    //private data
    caffe::shared_ptr<caffe::Net<float> > net_[4];
    int minFace_;
    float scale_;
    int stride_;
    float classThreshold_[3];
    float nmsThreshold_[3];
    float angleRange_;
    int period_;
    float trackThreshold_;
    float augScale_;
    cv::Scalar mean_;
    long global_id_; //Global ID incrementor
    std::vector<Window> preList_;
    int detectFlag_;
};

#endif
