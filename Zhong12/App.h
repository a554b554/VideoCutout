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
using namespace std;
using namespace cv;

class App{
public:
    App(string winName, string testpath):winName(winName),showIdx(0){
        DIR *dp;
        struct dirent *dirp;
        if((dp=opendir(testpath.c_str()))==NULL){
            perror("opendir error");
            free(dp);
            exit(1);
        }
        
        struct stat buf;
        while((dirp=readdir(dp))!=NULL){
            if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
                continue;
            string fname = testpath+dirp->d_name;
            imgs.push_back(imread(fname));
        }      
        closedir(dp);
    };
    string winName;
    void nextImg();
    void prevImg();
    void showImg();
private:
    int showIdx;
    vector<Mat> imgs;
    
};




#endif /* defined(__Zhong12__App__) */
