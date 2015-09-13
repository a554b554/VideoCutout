//
//  GlobalClassifier.cpp
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "GlobalClassifier.h"

void processGC(const Mat& img, const Mat& matte, const Mat& raw_dist, Mat& probmat, Mat& confmat){
    int64 t0 = getTickCount();

    
    
    probmat.create(img.rows, img.cols, CV_64FC1);
    confmat.create(img.rows, img.cols, CV_64FC1);
    vector<Vec3d> fSamples,bSamples;
    //create sample set.
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (fabs(raw_dist.at<float>(i,j))>30) {
                continue;
            }
            Vec3d color = img.at<Vec3b>(i,j);
            if (raw_dist.at<double>(i,j) < -5) { //prevent sampling error
                bSamples.push_back(color);
            }
            else if(raw_dist.at<double>(i,j) > 5){
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
        
        //debug
//        imshow("src", img);
//        imshow("probmat", probmat);
//        imshow("confmat", confmat);
//        waitKey(1);
    }
    
    
    cout<<"global classifier cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    
    //debug
//    probmat = probmat * 255;
//    confmat = confmat * 255;
//    probmat.convertTo(probmat, CV_8UC1);
//    confmat.convertTo(confmat, CV_8UC1);
//    imshow("probmat", probmat);
//    imshow("confmat", confmat);
//    imwrite("probmat.jpg", probmat);
//    imwrite("confmat.jpg", confmat);
//    imshow("matte", matte);
//    waitKey(0);
//    cout<<"glob: "<<probmat.at<double>(10, 371)<<endl;
}









