//
//  OpticalFlow.h
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__OpticalFlow__
#define __Zhong12__OpticalFlow__

#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include "common.h"
#include <opencv2/flann/flann.hpp>

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif


using namespace std;
using namespace cv;

class FrameProcessor{
public:
    virtual void process(Mat &input,Mat &ouput)=0;
};

class FeatureTracker :  public FrameProcessor{
    Mat gray;  //当前灰度图
    Mat gray_prev;  //之前的灰度图
    vector<Point2f> points[2];//前后两帧的特征点
    vector<Point2f> initial;//初始特征点
    vector<Point2f> features;//检测到的特征
    int max_count; //要跟踪特征的最大数目
    double qlevel; //特征检测的指标
    double minDist;//特征点之间最小容忍距离
    vector<uchar> status; //特征点被成功跟踪的标志
    vector<float> err; //跟踪时的特征点小区域误差和
public:
    FeatureTracker():max_count(500),qlevel(0.01),minDist(10.){}
    void process(Mat &frame,Mat &output);
    void detectFeaturePoint();
    bool addNewPoint();
    
    //若特征点在前后两帧移动了，则认为该点是目标点，且可被跟踪
    bool acceptTrackedPoint(int i);
    
    //画特征点
    void  drawTrackedPoint(Mat &frame,Mat &output);
};

class IFeatureMatcher {
public:
    virtual void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches) = 0;
    virtual std::vector<cv::KeyPoint> GetImagePoints(int idx) = 0;
};

class AbstractFeatureMatcher : public IFeatureMatcher {
protected:
    bool use_gpu;
public:
    AbstractFeatureMatcher(bool _use_gpu):use_gpu(_use_gpu) {}
};


class OFFeatureMatcher : public AbstractFeatureMatcher {
    std::vector<cv::Mat>& imgs;
    std::vector<std::vector<cv::KeyPoint> >& imgpts;
    
public:
    OFFeatureMatcher(bool _use_gpu,
                     std::vector<cv::Mat>& imgs_,
                     std::vector<std::vector<cv::KeyPoint> >& imgpts_);
    void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches);

    void registration(int idx_i, int idx_j, Mat& registrated_img);
    std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};






#endif /* defined(__Zhong12__OpticalFlow__) */
