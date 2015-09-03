//
//  CombinedClassifier.cpp
//  Zhong12
//
//  Created by DarkTango on 9/2/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "CombinedClassifier.h"

double featureVector::dist2(const featureVector &vec)const{
    return
    (ru-vec.ru)*(ru-vec.ru)+
    (rg-vec.rg)*(rg-vec.rg)+
    (rl-vec.rl)*(rl-vec.rl)+
    (rs-vec.rs)*(rs-vec.rs)+
    (e-vec.e)*(e-vec.e);
}

featureVector CombinedClassifier::getCorByID(long i){
    featureVector ans;
    ans.e = (double)(i%cSize)/(double)cSize;
    i/=cSize;
    ans.rs = (double)(i%cSize)/(double)cSize;
    i/=cSize;
    ans.rg = (double)(i%cSize)/(double)cSize;
    i/=cSize;
    ans.rl = (double)(i%cSize)/(double)cSize;
    i/=cSize;
    ans.ru = (double)(i%cSize)/(double)cSize;
    return ans;
}

void featureVector::print(){
    printf("%lf %lf %lf %lf %lf\n",ru,rl,rg,rs,e);
}

CombinedClassifier::CombinedClassifier(){
    init();
}

CombinedClassifier::CombinedClassifier(const string filepath){
    init();
}

void CombinedClassifier::init(){
    memset(fLattice, 0, interval*sizeof(double));
    memset(bLattice, 0, interval*sizeof(double));
}



void CombinedClassifier::train(const vector<Mat>& imgs, const vector<Mat>& mattes){
    for (int i = 1; i < imgs.size(); i++) {
        Mat valid(imgs[0].size(),CV_8UC1);
        valid.setTo(1);
        //process UDC
        Mat UDCprob,UDCconf;
        processUDC(imgs[i], mattes[i-1], valid, UDCprob, UDCconf);
        
        //process Local
        Mat localprob,localconf;
        processLC(imgs[i], mattes[i-1], localprob, localconf);
        
        //process Global
        Mat globalprob,globalconf;
        processGC(imgs[i], mattes[i-1], globalprob, globalconf);
        
        //process Shape
        Mat shapeprob,shapeconf;
        processSP(imgs[i], mattes[i-1], shapeprob, shapeconf);
        
        
        
        //only distance smaller than 30 are processed.
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        Mat matte_copy = mattes[i].clone();
        Mat img_copy = imgs[i].clone();
        findContours( matte_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        Mat raw_dist( imgs[0].size(), CV_32FC1 );
        for( int _j = 0; _j < imgs[0].rows; _j++ )
        {
            for( int _i = 0; _i < imgs[0].cols; _i++ )
            {
                raw_dist.at<float>(_j,_i) = pointPolygonTest( contours[0], Point2f(_i,_j), true );
            }
        }
        
        
        //set up feature vector
        for (int dx = 0; dx < imgs[0].rows; dx++) {
            for (int dy = 0; dy < imgs[0].cols; dy++) {
                
                float dist = fabs(raw_dist.at<float>(dx,dy));
                if (dist > 30) {
                    continue;
                }
                printf("dx:%d dy:%d\n",dx,dy);
                
                featureVector v;
                v.ru = 0.5 + UDCconf.at<double>(dx,dy)*(UDCprob.at<double>(dx,dy)-0.5);
                v.rl = 0.5 + localconf.at<double>(dx,dy)*(localprob.at<double>(dx,dy)-0.5);
                v.rg = 0.5 + globalconf.at<double>(dx,dy)*(globalprob.at<double>(dx,dy)-0.5);
                v.rs = 0.5 + shapeconf.at<double>(dx,dy)*(shapeprob.at<double>(dx,dy)-0.5);
                v.e = 0;
                if (mattes[i].at<uchar>(dx,dy)==0) {
                    addSample(v, false);
                }
                else{
                    addSample(v, true);
                }
            }
        }
    }
}

void CombinedClassifier::exportdata(){
    fstream f;
    f.open("data.txt");
    for (int i = 0; i < interval; i++) {
        f<<fLattice[i]<<" "<<bLattice[i]<<endl;
    }
    f.close();
}


void CombinedClassifier::addSample(featureVector v, bool addtoForeground){
    for (long i = 0; i < interval; i++) {
        featureVector current = getCorByID(i);
        double val = exp(-(current.dist2(v)/sigmad2));
        if (addtoForeground) {
            fLattice[i] += val;
        }
        else{
            bLattice[i] += val;
        }
        
    }
}