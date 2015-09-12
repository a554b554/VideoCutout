//
//  LocalClassifier.h
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__LocalClassifier__
#define __Zhong12__LocalClassifier__

#include <stdio.h>
#include "common.h"
#include "Classifier.h"
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

static const int K = 9;
static const int W = 10; //window size.

void processLC(const Mat& img, const Mat& matte, const Mat& raw_dist, Mat& probmat, Mat& confmat);


// label 1:=foreground 2:=background
class LocalClassifier{
public:
    LocalClassifier(const Vec3d& _color, const Mat& _traindata, const vector<int>& _label);
    double prob();
    double conf();
    static constexpr double sigma2 = 400.0;
private:
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knnMatches;
    const Mat& traindata;
    const Vec3d color;
    const vector<int>& label;
    double weight[K];
    double sum;
    double var;
};



#endif /* defined(__Zhong12__LocalClassifier__) */
