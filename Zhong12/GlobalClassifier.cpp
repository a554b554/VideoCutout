//
//  GlobalClassifier.cpp
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "GlobalClassifier.h"

void processGC(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat){
    vector<Vec3d> fSamples,bSamples;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d color = img.at<Vec3b>(i,j);
            if (matte.at<uchar>(i,j) == 0) {
                bSamples.push_back(color);
            }
            else{
                fSamples.push_back(color);
            }
        }
    }
    
    UDC udc(fSamples,bSamples);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d color = img.at<Vec3b>(i,j);
            probmat.at<double>(i,j) = udc.prob(color);
            confmat.at<double>(i,j) = udc.conf(color);
        }
    }
}