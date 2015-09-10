//
//  LocalClassifier.cpp
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "LocalClassifier.h"

LocalClassifier::LocalClassifier(const Vec3d& _color, const Mat& _traindata, const vector<int>& _label):color(_color),traindata(_traindata),label(_label){
    Vec3f __color = color;
    Mat sample(1, 3, CV_32FC1, (double*)&__color);
    //cout<<sample;
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
    double ans = sum*(1-4*var)/(double)K;
    return ans;
}

void processLC(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat){
    probmat.create(img.rows, img.cols, CV_64FC1);
    confmat.create(img.rows, img.cols, CV_64FC1);
    probmat.setTo(0);
    confmat.setTo(0);
    int64 t0 = getTickCount();
    
    
    //compute distance
    Mat raw_dist;
    computeRawDist(matte, raw_dist);
    //cout<<"cost for dist: "<<(getTickCount()-t1)/getTickFrequency()<<endl;
  //  cout<<raw_dist<<endl;
    
    for (int i = 0; i < img.rows; i++) {
        //printf("row: %d\n", i);
        for (int j = 0; j < img.cols; j++) {
            
            
            
            
            if (raw_dist.at<float>(i,j) < -20) { //which means outside the contour and distance greater than a threshold.
                probmat.at<double>(i,j) = 0;
                confmat.at<double>(i,j) = 1;
                continue;
            }
            
            int startx = (j-W)>=0?(j-W):0;
            int starty = (i-W)>=0?(i-W):0;
            if (j+W>=img.cols) {
                startx = img.cols-2*W-1;
            }
            if (i+W>=img.rows) {
                starty = img.rows-2*W-1;
            }
            
            Rect roi(startx,starty,2*W+1,2*W+1);
            Mat _traindata = img(roi).clone();
            Mat _label = matte(roi).clone();
            
//            debug
//            if (i>=110) {
//                Mat show = img.clone();
//                circle(show, Point(j,i), 2, CV_RGB(255, 0, 0));
//                rectangle(show, roi, CV_RGB(0, 255, 0));
//                imshow("test", show);
//                waitKey(0);
//            }
            
            //create label.
            vector<int> label;
            for (int dy = 0; dy < _label.rows; dy++) {
                for (int dx = 0; dx < _label.cols; dx++) {
                    if (_label.at<uchar>(dy,dx) == 0) {
                        label.push_back(0);
                    }
                    else{
                        label.push_back(1);
                    }
                }
            }
            
            //create train data.
            _traindata = _traindata.reshape(1,_traindata.cols*_traindata.rows);
            _traindata.convertTo(_traindata, CV_32FC1);
            //compute probability and confidence
            Vec3d color = img.at<Vec3b>(i,j);
            LocalClassifier lc(color, _traindata, label);
            probmat.at<double>(i,j) = lc.prob();
            confmat.at<double>(i,j) = lc.conf();
            
            
        }
    }
    cout<<"local classify cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    
    //debug
//    imshow("prob", probmat);
//    imshow("conf", confmat);
//    imshow("img", img);
//    imshow("matte", matte);
//    probmat=probmat*255;
//    probmat.convertTo(probmat, CV_8UC1);
//    threshold(probmat, probmat, 10, 255, CV_THRESH_BINARY);
//    Mat show;
//    img.copyTo(show, probmat);
//    imshow("final", show);
//    waitKey(0);
    
}










