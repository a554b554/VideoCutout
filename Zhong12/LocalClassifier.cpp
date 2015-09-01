//
//  LocalClassifier.cpp
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "LocalClassifier.h"

LocalClassifier::LocalClassifier(const Vec3d _color, const Mat& _traindata, const vector<int>& _label):color(_color),traindata(_traindata),label(_label){
    Vec3f __color = color;
    Mat sample(1, 3, CV_32FC1, (double*)&__color);
    cout<<sample;
    matcher.knnMatch(sample, traindata, knnMatches, K);
    for (int i = 0; i < K; i++) {
        weight[i] = exp(-(knnMatches[0][i].distance * knnMatches[0][i].distance)/sigma2);
    }
    sum = 0;
    for (int i = 0; i < K; i++) {
        sum += weight[i];
    }
    
    //compute variance.
    var = variance(label);
}

double LocalClassifier::prob(){
    double ans = 0;
    for (int i = 0; i < K; i++) {
        ans+=weight[i]*label[knnMatches[0][i].trainIdx];
    }
    return ans/sum;
}

double LocalClassifier::conf(){
    return sum*(1-4*var)/(double)K;
}

void processLC(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat){
    probmat.create(img.rows, img.cols, CV_64FC1);
    confmat.create(img.rows, img.cols, CV_64FC1);
    probmat.setTo(0);
    confmat.setTo(0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            
        }
    }
}