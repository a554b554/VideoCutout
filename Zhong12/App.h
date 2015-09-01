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

using namespace std;
using namespace cv;

class App{
public:
    App(string winName, string testpath, string dirname);
    string winName;
    void nextImg();
    void prevImg();
    void showImg();
    void calcOpticalFlows();
    void changeShowState();
    void testUDC();
    void testLocal();
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
};




#endif /* defined(__Zhong12__App__) */
