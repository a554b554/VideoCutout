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

featureVector featureVector::operator*(double num){
    featureVector ans;
    ans.ru = this->ru*num;
    ans.rl = this->rl*num;
    ans.rg = this->rg*num;
    ans.rs = this->rs*num;
    ans.e = this->e*num;
    return ans;
}

int CombinedClassifier::getNearestVectorID(featureVector v){
    int bound = cSize-1;
    int ru = (int)(v.ru*cSize+0.5)<=bound?(int)(v.ru*cSize+0.5):bound;
    int rl = (int)(v.rl*cSize+0.5)<=bound?(int)(v.rl*cSize+0.5):bound;
    int rg = (int)(v.rg*cSize+0.5)<=bound?(int)(v.rg*cSize+0.5):bound;
    int rs = (int)(v.rs*cSize+0.5)<=bound?(int)(v.rs*cSize+0.5):bound;
    int e = (int)(v.e*cSize+0.5)<=bound?(int)(v.e*cSize+0.5):bound;
    
    
    return ru*cSize*cSize*cSize*cSize+
    rl*cSize*cSize*cSize+
    rg*cSize*cSize+
    rs*cSize+
    e;
}

void featureVector::print(){
    printf("%lf %lf %lf %lf %lf\n",ru,rl,rg,rs,e);
}

CombinedClassifier::CombinedClassifier(){
    init();
}

CombinedClassifier::CombinedClassifier(const string filepath){
    init();
    fstream f;
    f.open(filepath);
    for (int i = 0; i < interval; i++) {
        f>>fLattice[i]>>bLattice[i];
    }
    f.close();
}

CombinedClassifier::CombinedClassifier(const vector<string>& filepath){
    init();
    
    for (int i = 0; i < filepath.size(); i++) {
        fstream f;
        f.open(filepath[i]);
        if (!f.is_open()) {
            cerr<<"no such file exist: "<<filepath[i]<<"!"<<endl;
            exit(1);
        }
      //  printf(filepath[i].c_str());
        for (int j = 0; j < interval; j++) {
            double fl,bl;
            f>>fl>>bl;
        //    printf("%lf %lf\n",fl,bl);
            fLattice[j]+=fl;
            bLattice[j]+=bl;
        }
        f.close();
    }
    
}

void CombinedClassifier::init(){
    memset(fLattice, 0, interval*sizeof(double));
    memset(bLattice, 0, interval*sizeof(double));
}



void CombinedClassifier::train(const vector<Mat>& imgs, const vector<Mat>& mattes_gt, const vector<Mat>& remats, const vector<Mat>& mattes_warped){
    for (int i = 1; i < imgs.size(); i++) {
        printf("img: %d\n",i);
        int64 t0 = getTickCount();
        Mat valid(imgs[0].size(),CV_8UC1);
        valid.setTo(1);
        Mat raw_dist;
        computeRawDist(mattes_warped[i-1], raw_dist);
        //process UDC
        Mat UDCprob,UDCconf;
        processUDC(imgs[i], mattes_warped[i-1], raw_dist, UDCprob, UDCconf);
        
        //process Local
        Mat localprob,localconf;
        processLC(imgs[i], mattes_warped[i-1], raw_dist, localprob, localconf);
        
        //process Global
        Mat globalprob,globalconf;
        processGC(imgs[i], mattes_warped[i-1], raw_dist, globalprob, globalconf);
        
        //process Shape
        Mat shapeprob,shapeconf;
        processSP(imgs[i], mattes_warped[i-1], raw_dist, shapeprob, shapeconf);
        
        Mat errordensity;
        processRegistraionError(remats[i-1], errordensity);
        
        
        
//        FileStorage fs("tmp.xml", FileStorage::WRITE);
//        fs<<"UDCprob"<<UDCprob;
//        fs<<"UDCconf"<<UDCconf;
//        fs<<"localprob"<<localprob;
//        fs<<"localconf"<<localconf;
//        fs<<"globalprob"<<globalprob;
//        fs<<"globalconf"<<globalconf;
//        fs<<"shapeprob"<<shapeprob;
//        fs<<"shapeconf"<<shapeconf;
//        fs<<"errordensity"<<errordensity;
//        fs.release();
        
        
//        debug show
//        imshow("target", mattes_gt[i]);
//        imshow("warp", mattes_warped[i-1]);
//        imshow("image", imgs[i]);
//        imshow("UDCp", UDCprob);
//        imshow("UDCc", UDCconf);
//        imshow("localp", localprob);
//        imshow("localc", localconf);
//        imshow("globalp", globalprob);
//        imshow("globalc", globalconf);
//        imshow("shapep", shapeprob);
//        imshow("shapec", shapeconf);
//        imshow("rege", errordensity);
//        waitKey(0);
        
        
        
        //only distance smaller than 30 are processed.
        Mat raw_dist_gt;
        computeRawDist(mattes_gt[i], raw_dist_gt);
        
        
        //set up feature vector
        for (int dx = 0; dx < imgs[0].rows; dx++) {
            for (int dy = 0; dy < imgs[0].cols; dy++) {
                
                float dist = fabs(raw_dist_gt.at<float>(dx,dy));
                if (dist > 30) {
                    continue;
                }
                //printf("dx:%d dy:%d\n",dx,dy);
                
                featureVector v;
                v.ru = 0.5 + UDCconf.at<double>(dx,dy)*(UDCprob.at<double>(dx,dy)-0.5);
                v.rl = 0.5 + localconf.at<double>(dx,dy)*(localprob.at<double>(dx,dy)-0.5);
                v.rg = 0.5 + globalconf.at<double>(dx,dy)*(globalprob.at<double>(dx,dy)-0.5);
                v.rs = 0.5 + shapeconf.at<double>(dx,dy)*(shapeprob.at<double>(dx,dy)-0.5);
                v.e = errordensity.at<double>(dx,dy);
                
                if (mattes_gt[i].at<uchar>(dx,dy)==0) {
                    addSample(v, false);
                }
                else{
                    addSample(v, true);
                }
            }
        }
        
        cout<<"train image cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    }
}

void CombinedClassifier::exportdata(){
    fstream f;
    f.open("../data.txt");
    for (int i = 0; i < interval; i++) {
        f<<fLattice[i]<<" "<<bLattice[i]<<endl;
    }
    f.close();
}

static const int gridsize = 2;

void CombinedClassifier::addSample(featureVector _v, bool addtoForeground){
    long id = getNearestVectorID(_v);
    featureVector v = this->getCorByID(id);
//    _v.print();
//    v.print();
    
    int rustart = ((int)(v.ru*cSize)-gridsize)>=0?(int)(v.ru*cSize)-gridsize:0;
    int ruend = ((int)(v.ru*cSize)+gridsize)<=cSize?(int)(v.ru*cSize)+gridsize:cSize;
    int rlstart = ((int)(v.rl*cSize)-gridsize)>=0?(int)(v.rl*cSize)-gridsize:0;
    int rlend = ((int)(v.rl*cSize)+gridsize)<=cSize?(int)(v.rl*cSize)+gridsize:cSize;
    int rgstart = ((int)(v.rg*cSize)-gridsize)>=0?(int)(v.rg*cSize-gridsize):0;

    int rgend = ((int)(v.rg*cSize)+gridsize)<=cSize?(int)(v.rg*cSize)+gridsize:cSize;
    int rsstart = ((int)(v.rl*cSize)-gridsize)>=0?(int)(v.rl*cSize)-gridsize:0;
    int rsend = ((int)(v.rl*cSize)+gridsize)<=cSize?(int)(v.rl*cSize)+gridsize:cSize;
    int estart = ((int)(v.e*cSize)-gridsize)>=0?(int)(v.e*cSize)-gridsize:0;
    int eend = ((int)(v.e*cSize)+gridsize)<=cSize?(int)(v.e*cSize)+gridsize:cSize;

    for (int ru = rustart; ru < ruend; ru++) {
        for (int rl = rlstart; rl < rlend; rl++) {
            for (int rg = rgstart; rg < rgend; rg++) {
                for (int rs = rsstart; rs < rsend; rs++) {
                    for (int e = estart; e < eend; e++) {
                        
                        long id = e+
                        rs*cSize+
                        rg*cSize*cSize+
                        rl*cSize*cSize*cSize+
                        ru*cSize*cSize*cSize*cSize;
                        featureVector current = getCorByID(id);
                        //current.print();
                        

                        
                        double val = exp(-(current.dist2(v)/sigmad2));
                        if (isnan(val)) {
                            printf("nan!");
                            val = 0;
                        }
                        // cout<<val<<endl;
                        if (addtoForeground) {
                            fLattice[id] += val;
                            //cout<<"f"<<id<<": "<<fLattice[id]<<endl;
                        }
                        else{
                            bLattice[id] += val;
                            //cout<<"b"<<id<<": "<<bLattice[id]<<endl;
                        }
                    }
                }
            }
        }
    }
//    for (long i = 0; i < interval; i++) {
//        featureVector current = getCorByID(i);
//        current.print();
//        double val = exp(-(current.dist2(v)/sigmad2));
//        if (addtoForeground) {
//            fLattice[i] += val;
//        }
//        else{
//            bLattice[i] += val;
//        }
//        
//    }
//
    
}



double CombinedClassifier::prob(featureVector v){
    int id = getNearestVectorID(v);
    return fLattice[id]/(fLattice[id]+bLattice[id]);
}

static const double ep = 5;
double CombinedClassifier::conf(featureVector v){
    int id = getNearestVectorID(v);
    return fabs(fLattice[id]-bLattice[id])/(fLattice[id]+bLattice[id]+ep);
}






















