//
//  RegistrationError.cpp
//  Zhong12
//
//  Created by DarkTango on 9/2/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "RegistrationError.h"
static const double sigmae2 = 400;
static const int winstep = 3;
static constexpr double winsize = (2*winstep+1) * (2*winstep+1);

void processRegistraionError(const Mat& errormat, Mat& errordensity){
    errordensity.create(errormat.size(), CV_64FC1);
    int64 t0 = getTickCount();
    
    for (int i = 0; i < errormat.rows; i++) {
        for (int j = 0; j < errormat.cols; j++) {
            int xstart = (j-winstep)>0?(j-winstep):0;
            int xend = (j+winstep)<errormat.cols?(j+winstep):errormat.cols-1;
            int ystart = (i-winstep)>0?(i-winstep):0;
            int yend = (i+winstep)<errormat.rows?(i+winstep):errormat.rows-1;
            
            double ans = 0;
            for (int dy = ystart; dy <= yend; dy++) { //7x7 window
                for (int dx = xstart; dx <= xend; dx++) {
                    int tmp = (int)errormat.at<uchar>(dy,dx);
                    ans += tmp*tmp;
                }
            }
            
            errordensity.at<double>(i,j) = 1 - exp(-(ans/winsize)/sigmae2);
        }
    }
    
    cout<<"registration error cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
}