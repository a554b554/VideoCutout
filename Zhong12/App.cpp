//
//  App.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "App.h"

void App::showImg(){
    switch (currentShowState) {
        case SHOW_ORIGIN:
            printf("show origin\n");
            imshow(winName, imgs[showIdx]);
            break;
        case SHOW_OPTICALFLOW:
            printf("show optical flow\n");
            imshow(winName, optflows[showIdx]);
            break;
        default:
            break;
    }
}

void App::nextImg(){
    showIdx = (showIdx+1) % imgs.size();
    showImg();
}

void App::prevImg(){
    showIdx = (showIdx - 1)>=0? (showIdx-1):(imgs.size()-1);
    showImg();
}

void App::calcOpticalFlows(){
    printf("calculating optical flow....\n");
    optflows.resize(imgs.size());
//    for (int i = 0; i < imgs.size(); i++) {
//        tracker->process(imgs[i], optflows[i]);
//    }
    vector<vector<KeyPoint>> img_pts;
    OFFeatureMatcher* matcher = new OFFeatureMatcher(true, imgs, img_pts);
    vector<vector<DMatch>> matches;
    matches.resize(imgs.size()-1);
    matcher->registration(0, 3, optflows[0]);
    for (int i = 0; i < imgs.size() - 1; i++) {
        matcher->MatchFeatures(i, i+1, &matches[i], optflows[i]);
    }
    optflows[imgs.size()-1] = imgs[imgs.size()-1].clone();
    
    
    
    printf("calculate finished!\n");
}

void App::changeShowState(){
    printf("currentstate:%d\n",currentShowState);
    currentShowState = (currentShowState+1) % 2;
    showImg();
}