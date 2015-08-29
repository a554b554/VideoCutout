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
#include <opencv2/calib3d/calib3d.hpp>

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif


using namespace std;
using namespace cv;


class IFeatureMatcher {
public:
    virtual void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches,Mat& output = noArray().getMatRef()) = 0;
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
    void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches,Mat& output);

    void registration(int idx_i, int idx_j, Mat& registrated_img);
    std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};






#endif /* defined(__Zhong12__OpticalFlow__) */
