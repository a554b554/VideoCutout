//
//  ShapePrior.cpp
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "ShapePrior.h"
void processSP(const Mat& img, const Mat& matte, const Mat& raw_dist, Mat& probmat, Mat& confmat){
    int64 t0 = getTickCount();
    probmat = matte.clone();
    probmat.convertTo(probmat, CV_64FC1);
    probmat = probmat/255;
    confmat.create(img.size(), CV_64FC1);
    
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            double dist = raw_dist.at<float>(i,j);
            confmat.at<double>(i,j) = 1 - exp(-(dist*dist)/sigmas2);
        }
    }
    
    
    
    //debug
//    drawContours(img_copy, contours, 0, Scalar(255,0,0));
//    imshow("d", img_copy);
//    imshow("probmat", probmat);
//    imshow("confmat", confmat);
//    waitKey(0);
    cout<<"shape prior cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;

}