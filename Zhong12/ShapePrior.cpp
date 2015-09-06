//
//  ShapePrior.cpp
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "ShapePrior.h"
void processSP(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat){
    int64 t0 = getTickCount();
    probmat = matte.clone();
    probmat.convertTo(probmat, CV_64FC1);
    probmat = probmat/255;
    confmat.create(img.size(), CV_64FC1);
    
    vector<vector<Point> > contours; vector<Vec4i> hierarchy;
    Mat matte_copy = matte.clone();
    Mat img_copy = img.clone();
    
    findContours( matte_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    Mat raw_dist( img.size(), CV_32FC1 );
    
    for( int j = 0; j < img.rows; j++ )
    {
        for( int i = 0; i < img.cols; i++ )
        {
            raw_dist.at<float>(j,i) = pointPolygonTest( contours[0], Point2f(i,j), true );
        }
    }
    
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