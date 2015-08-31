//
//  UDC.h
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__UDC__
#define __Zhong12__UDC__

#include <stdio.h>
#include "common.h"
#include "GMM.h"
#include "Classifier.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
const int RectHeight = 45;
const int range = 20;

void processUDC(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat);

void processUDCRect(const Mat& img_rect, const Mat& matte_rect, Mat& probmat, Mat& confmat);

void getRectangle(const Mat& matte, int direction, vector<Rect>& rects);

void processUDC(const Mat& img, const Mat& matte, const Mat& valid, Mat& probmat, Mat& confmat);


class UDC : public Classifier{
private:
    GMM* fGMM;
    GMM* bGMM;
    vector<int> fLabel;
    vector<int> bLabel;
public:
    static constexpr double epi = 0.001;
    UDC(const vector<Vec3d>& fgdSamples, const vector<Vec3d>& bgdSamples);
    double prob(const Vec3d color) const;
    double conf(const Vec3d color) const;
    ~UDC();
};



#endif /* defined(__Zhong12__UDC__) */
