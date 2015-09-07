//
//  UDC.cpp
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "UDC.h"

UDC::UDC(const vector<Vec3d>& fgdSamples, const vector<Vec3d>& bgdSamples){
    Mat empty;
    fGMM = new GMM(empty);
    bGMM = new GMM(empty);
    fGMM->learning(fgdSamples);
    bGMM->learning(bgdSamples);
}

double UDC::prob(const Vec3d color)const{
    return (*fGMM)(color)/((*fGMM)(color)+(*bGMM)(color));
}

double UDC::conf(const Vec3d color)const{
    double term1 = fabs((*fGMM)(color) - (*bGMM)(color))/((*fGMM)(color) + (*bGMM)(color) + epi);
    double qx = prob(color)*(fGMM->quantity(color, true))+(1-prob(color))*(bGMM->quantity(color, false));
    return term1;
}

UDC::~UDC(){
    delete fGMM;
    delete bGMM;
}

void processUDC(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat){
    

}

void processUDCRect(const Mat& img_rect, const Mat& matte_rect, Mat& probmat, Mat& confmat){
    vector<Vec3d> fSamples,bSamples;
    int rows = matte_rect.rows;
    int cols = matte_rect.cols;
    
    //learning GMM

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matte_rect.at<uchar>(i,j) == 0) {
                bSamples.push_back(static_cast<Vec3d>(img_rect.at<Vec3b>(i,j)));
            }
            else if (matte_rect.at<uchar>(i,j) == 255){
                fSamples.push_back(static_cast<Vec3d>(img_rect.at<Vec3b>(i,j)));
            }
        }
    }
    

    //compute confidence map and probability map.
    probmat.create(rows, cols, CV_64FC1);
    confmat.create(rows, cols, CV_64FC1);
    if (fSamples.size()<10||bSamples.size()<10) {
        probmat.setTo(0);
        confmat.setTo(0);
        return;
    }
    UDC udc(fSamples,bSamples);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3d color = img_rect.at<Vec3b>(i,j);
            probmat.at<double>(i,j) = udc.prob(color);
            confmat.at<double>(i,j) = udc.conf(color);
        }
    }
}

//0:=0, 1:=45, 2:=90, 3:=135 degree
void getRectangle(const Mat& matte, int direction, vector<Rect>& rects){
    // rect is the RotatedRect (I got it from a contour...)
    RotatedRect rect = RotatedRect(Point2f(100,100), Size2f(100,50), 30);
    // matrices we'll use
    Mat M, rotated, cropped;
    // get angle and size from the bounding box
    float angle = rect.angle;
    Size rect_size = rect.size;
    if (rect.angle < -45.) {
        angle += 90.0;
        swap(rect_size.width, rect_size.height);
    }
    // get the rotation matrix
    M = getRotationMatrix2D(rect.center, angle, 1.0);
    // perform the affine transformation
    warpAffine(matte  , rotated, M, matte.size(), INTER_CUBIC);
    // crop the resulting image
    getRectSubPix(rotated, rect_size, rect.center, cropped);
    imshow("rot", rotated);
    imshow("crop", cropped);
    waitKey(0);
}

void processUDC(const Mat& img, const Mat& matte, const Mat& valid, Mat& probmat, Mat& confmat){
    int64 t0 = getTickCount();
    int step = RectHeight*2/3;
    Mat lastconf,lastprob;
    //compute bounding box.
    probmat.create(img.rows, img.cols, CV_64FC1);
    confmat.create(img.rows, img.cols, CV_64FC1);
    probmat.setTo(0);
    confmat.setTo(1);
    
    for (int base = 0; base < img.rows; base+=step) {
        int start = base;
        int end = (base + RectHeight)>img.rows?img.rows:(base+RectHeight);
        int rectstart=0,rectend=0;
        bool find = false;
        
        //forward scan, find rectstart.
        for (int j = 0; j < img.cols; j++) {
            if(find){
                break;
            }
            for (int i = start; i < end; i++) {
                
                if (valid.at<uchar>(i,j)==0) {
                    break;
                }
                
                if (matte.at<uchar>(i,j)!=0) {
                    rectstart = (j-range>0)?(j-range):0;
                    find = true;
                    break;
                }
            }
        }
        find = false;
        
        //backward scan, find rectend.
        for (int j = img.cols-1; j>=0; j--) {
            if (find) {
                break;
            }
            for (int i = start; i < end; i++) {
                if (valid.at<uchar>(i,j)==0) {
                    break;
                }
                if (matte.at<uchar>(i,j)!=0) {
                    rectend = (j+range<img.cols)?(j+range):img.cols-1;
                    find = true;
                    break;
                }
            }
        }
        
        
        
        if (find) {
            //show rectangle
            Rect rec(rectstart, base, rectend-rectstart, base+RectHeight>=img.rows?img.rows-1-base:RectHeight);
            
            //debug
//            Mat showrec = img.clone();
//            rectangle(showrec, rec, Scalar(255,0,0));
//            imshow("show rec", showrec);
            
            
            //compute probability and confidence
            Mat prob,conf;
            processUDCRect(img(rec), matte(rec), prob, conf);
           
//            if (!lastconf.empty()) {
//                for (int dy = 0; dy < RectHeight/3; dy++) {
//                    for (int dx = 0; dx < conf.cols; dx++) {
//                        if (conf.at<double>(dy, dx) < lastconf.at<double>(dy+2*RectHeight/3,dx)) {
//                            conf.at<double>(dy, dx) = lastconf.at<double>(dy+2*RectHeight/3, dx);
//                            prob.at<double>(dy, dx) = lastprob.at<double>(dy+2*RectHeight/3, dx);
//                        }
//                    }
//                }
//            }
//            probmat(rec).setTo(prob, valid(rec));
//            confmat(rec).setTo(conf, valid(rec));
            prob.copyTo(probmat(rec), valid(rec));
            conf.copyTo(confmat(rec), valid(rec));
            if (base+step<img.rows) {
                prob.copyTo(lastprob);
                conf.copyTo(lastconf);
            }
            
//            imshow("prob", prob);
//            imshow("conf", conf);
//            waitKey(0);
        }
        if (base + RectHeight > img.rows) {
            break;
        }
    }
    
    
    //debug
//    imshow("finalprob", probmat);
//    imshow("finalconf", confmat);
//    waitKey(0);
    cout<<"UDC cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
}









