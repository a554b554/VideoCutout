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

double UDC::prob(const Vec3d& color)const{
    return (*fGMM)(color)/((*fGMM)(color)+(*bGMM)(color));
}

double UDC::conf(const Vec3d& color)const{
    double term1 = fabs((*fGMM)(color) - (*bGMM)(color))/((*fGMM)(color) + (*bGMM)(color) + epi);
    double qx = prob(color)*(fGMM->quantity(color, true))+(1-prob(color))*(bGMM->quantity(color, false));
    return term1;
}

UDC::~UDC(){
    delete fGMM;
    delete bGMM;
}

void getbestmap(const vector<Mat>& probs, const vector<Mat>& confs, Mat& bestprob, Mat& bestconf){
    bestprob.create(probs[0].size(), CV_64FC1);
    bestconf.create(confs[0].size(), CV_64FC1);
    
    for (int i = 0; i < bestconf.rows; i++) {
        for (int j = 0; j < bestconf.cols; j++) {
            //find best
            int idx = 0;
            double c = confs[0].at<double>(i,j);
            for (int cc = 0; cc < probs.size(); cc++) {
                if (confs[cc].at<double>(i,j) > c) {
                    c = confs[cc].at<double>(i,j);
                    idx = cc;
                }
            }
            
            bestconf.at<double>(i,j) = c;
            bestprob.at<double>(i,j) = probs[idx].at<double>(i,j);
        }
    }
    
}

void rotateBack(const Mat& src, const Mat& valid, Point2f center, Mat& dst, Size dsize, bool is45){
    Mat rot;
    if (is45) {
        rot = getRotationMatrix2D(center, -45, 1);
    }
    else{
        rot = getRotationMatrix2D(center, -135, 1);
    }

    dst.create(dsize, CV_64FC1);
    Mat src_copy = src.clone();
    Mat valid_copy = valid.clone();
    warpAffine(src_copy, src_copy, rot, src_copy.size());
    warpAffine(valid_copy, valid_copy, rot, valid_copy.size());
    
   

    Rect roi(src.cols/2-dsize.width/2,src.rows/2-dsize.height/2,dsize.width,dsize.height);
    src_copy(roi).copyTo(dst);
    
    
    
    //debug
//    imshow("src_copy", src_copy);
//    imshow("dst", dst);
//    waitKey(0);
    
}



void processUDC(const Mat& img, const Mat& matte, const Mat& raw_dist, Mat& probmat, Mat& confmat){

    
    //process degree 0
    Mat valid(img.size(),CV_8UC1),prob0,conf0;
    vector<Mat> probs,confs;
    valid.setTo(1);
    processUDC(img, matte, valid, raw_dist,prob0, conf0);
    probs.push_back(prob0);
    confs.push_back(conf0);
    //process degree 90
    Mat valid_t = valid.t();
    Mat img_t = img.t();
    Mat matte_t = matte.t();
    Mat raw_dist_t = raw_dist.t();
    Mat prob90,conf90;
    processUDC(img_t, matte_t, valid_t, raw_dist_t, prob90, conf90);
    prob90 = prob90.t();
    conf90 = conf90.t();
    probs.push_back(prob90);
    confs.push_back(conf90);
    
    
    
    //process degree 45
    int length = max(max(img.cols, img.rows), (int)(sqrt(2)*(img.cols+img.rows)/2));
    
    
    Mat _img45(length,length,CV_8UC3);
    Mat _valid45(length,length,CV_8UC1);
    Mat _matte45(length,length,CV_8UC1);
    Mat _raw_dist45(length,length,CV_64FC1);
    Mat _prob45,_conf45,prob45,conf45;
    
    _img45.setTo(0);
    _valid45.setTo(0);
    _matte45.setTo(0);
    _raw_dist45.setTo(0);
    
    int offside = length - max(img.cols, img.rows);
    offside = offside>0?offside:0;
    Rect roi(offside/2,length/2-img.rows/2,img.cols,img.rows);
    
    
    //warp image 45 degree
    Point2f center(length/2,length/2);
    Mat rot = getRotationMatrix2D(center, 45, 1);
    img.copyTo(_img45(roi));
    matte.copyTo(_matte45(roi));
    _valid45(roi).setTo(255);
    raw_dist.copyTo(_raw_dist45(roi));
    warpAffine(_matte45, _matte45, rot, _matte45.size());
    warpAffine(_img45, _img45, rot, _img45.size());
    warpAffine(_valid45, _valid45, rot, _img45.size());
    warpAffine(_raw_dist45, _raw_dist45, rot, _raw_dist45.size());
    
    processUDC(_img45, _matte45, _valid45, _raw_dist45, _prob45, _conf45);
    
    
    rotateBack(_prob45, _valid45, center, prob45, img.size(), true);
    rotateBack(_conf45, _valid45, center, conf45, img.size(), true);
    probs.push_back(prob45);
    confs.push_back(conf45);
    
    
    //process degree 135
    Mat _img135(length,length,CV_8UC3);
    Mat _valid135(length,length,CV_8UC1);
    Mat _matte135(length,length,CV_8UC1);
    Mat _raw_dist135(length,length,CV_64FC1);
    Mat _prob135,_conf135,prob135,conf135;
    
    _img135.setTo(0);
    _valid135.setTo(0);
    _matte135.setTo(0);
    _raw_dist135.setTo(0);
    
    offside = offside>0?offside:0;
    
    
    //warp image 45 degree
    rot = getRotationMatrix2D(center, 135, 1);
    img.copyTo(_img135(roi));
    matte.copyTo(_matte135(roi));
    raw_dist.copyTo(_raw_dist135(roi));
    _valid135(roi).setTo(255);
    warpAffine(_matte135, _matte135, rot, _matte135.size());
    warpAffine(_img135, _img135, rot, _img135.size());
    warpAffine(_valid135, _valid135, rot, _img135.size());
    warpAffine(_raw_dist135, _raw_dist135, rot, _raw_dist135.size());
    
    processUDC(_img135, _matte135, _valid135, _raw_dist135, _prob135, _conf135);
    
    
    rotateBack(_prob135, _valid135, center, prob135, img.size(), false);
    rotateBack(_conf135, _valid135, center, conf135, img.size(), false);
    probs.push_back(prob135);
    confs.push_back(conf135);

    
    
    getbestmap(probs, confs, probmat, confmat);
    
    //debug
//    imshow("prob0", prob0);
//    imshow("conf0", conf0);
//    imshow("prob90", prob90);
//    imshow("conf90", conf90);
//    imshow("prob45", prob45);
//    imshow("conf45", conf45);
//    imshow("prob135", prob135);
//    imshow("conf135", conf135);
//    imshow("prob", probmat);
//    imshow("conf", confmat);
//    waitKey(0);
//    destroyAllWindows();
    

}





void processUDCRect(const Mat& img_rect, const Mat& matte_rect, const Mat& raw_dist_rect, Mat& probmat, Mat& confmat){
    vector<Vec3d> fSamples,bSamples;
    int rows = matte_rect.rows;
    int cols = matte_rect.cols;
    
//    learning GMM by matte

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(raw_dist_rect.at<double>(i,j)) < 5) { //prevent sampling error
                continue;
            }
            if (matte_rect.at<uchar>(i,j) == 0) {
                bSamples.push_back(static_cast<Vec3d>(img_rect.at<Vec3b>(i,j)));
            }
            else if (matte_rect.at<uchar>(i,j) == 255){
                fSamples.push_back(static_cast<Vec3d>(img_rect.at<Vec3b>(i,j)));
            }
        }
    }
    
    //learning GMM by kmeans
//    Mat lab;
//    Mat sample = img_rect.clone();
//    Mat _matte = matte_rect.clone();
//    _matte.create(matte_rect.cols*matte_rect.rows, 1, CV_8UC1);
//    sample.create(img_rect.cols*img_rect.rows, 1, CV_8UC3);
//    sample.convertTo(sample, CV_32FC1);
//    kmeans(sample, 2, lab, TermCriteria( CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
//    
//    
//    int count=0;
//    for (int i = 0; i < lab.rows; i++) {
//        if (_matte.at<uchar>(i,0) != 0) {
//            if (lab.at<int>(i,0) == 0) {
//                count++;
//            }
//            else{
//                count--;
//            }
//        }
//    }
//    int bl = count>0?1:0;
//    for (int i = 0; i < sample.rows; i++) {
//        if (lab.at<int>(i,0) == bl) {
//            bSamples.push_back(sample.at<Vec3f>(i,0));
//        }
//        else{
//            fSamples.push_back(sample.at<Vec3f>(i,0));
//        }
//    }
    

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

void processUDC(const Mat& img, const Mat& matte, const Mat& valid,const Mat& raw_dist, Mat& probmat, Mat& confmat){
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
                    //base += step;
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
            processUDCRect(img(rec), matte(rec), raw_dist(rec), prob, conf);
//            imshow("prob", probmat);
//            imshow("conf", confmat);
//            waitKey(0);
            
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









