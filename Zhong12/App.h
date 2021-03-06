//
//  App.h
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__App__
#define __Zhong12__App__

#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "OpticalFlow.h"
#include "UDC.h"
#include "LocalClassifier.h"
#include "GlobalClassifier.h"
#include "ShapePrior.h"
#include "CombinedClassifier.h"
#include "common.h"

#define MAGIC_NUMBER_BGR 2568


using namespace std;
using namespace cv;

void loadimage(string dirname, vector<Mat>& imgs, vector<Mat>& mattes, int code = MAGIC_NUMBER_BGR);
void loadmatte(string dirname, vector<Mat>& mattes);
void parse(string filepath, vector<string>& content);


class App{
public:
    App(string winName, string testpath, string dirname); // processing
    App(string winName, string filelistpath); //training
    string winName;
    void nextImg();
    void prevImg();
    void showImg();
    void calcOpticalFlows();
    void changeShowState();
    void start(const vector<string>& trained); //compute by ground truth
    void start2(const vector<string>& trained); //compute by previous frame
    void start3(const vector<string>& trained); //begin refinement
    void creategroundtruth();
    void computeOpitcalFlow(const Mat& srcimg, const Mat& dstimg, const Mat& srcmatte, Mat& warped_matte, Mat& warped_img);
    void reverse();
    
    void refinement(vector<Mat>& right, vector<Mat>& left);
    
    //unit test
    void testUDC();
    void testLocal();
    void testGlobal();
    void testShape();
    void testlearn();
    void testre();
    void exportimg(const vector<Mat>& imgs, string path);
    void maskbypreviousframe();
    
private:
    enum showState{
        SHOW_ORIGIN,
        SHOW_WARP,
        SHOW_MATTE_GT,
        SHOW_MATTE_WARP
    };
    int currentShowState;
    int showIdx;
    vector<Mat> imgs;
    vector<Mat> mattes;
    vector<Mat> warped_imgs;
    vector<Mat> warped_mattes;
    vector<Mat> remats; //uchar
    
    vector<Mat> output_probs;
    vector<Mat> output_confs;
    vector<Mat> final;
    
    vector<Mat> forward;
    vector<Mat> backward;
    
    CombinedClassifier* classifier;
    void clear(); //clear all storaged image in App.
};




#endif /* defined(__Zhong12__App__) */
