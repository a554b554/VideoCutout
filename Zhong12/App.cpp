//
//  App.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "App.h"


void loadimage(string dirname, vector<Mat>& imgs, vector<Mat>& mattes){
    string path = dirname + "/";
    string alpha = dirname + "_alpha/";
    
    
    //load image
    DIR *dp;
    struct dirent *dirp;
    if((dp=opendir(path.c_str()))==NULL){
        perror("opendir error");
        free(dp);
        exit(1);
    }
    
    struct stat buf;
    while((dirp=readdir(dp))!=NULL){
        if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
            continue;
        string fname = path+dirp->d_name;
        imgs.push_back(imread(fname));
    }
    closedir(dp);
    
    //load mattes.
    if((dp=opendir(alpha.c_str()))==NULL){
        perror("opendir error");
        free(dp);
        exit(1);
    }
    
    while((dirp=readdir(dp))!=NULL){
        if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
            continue;
        string fname = alpha+dirp->d_name;
        mattes.push_back(imread(fname,0));
        //        imshow("matte", mattes[mattes.size()-1]);
        //        waitKey(0);
    }
    closedir(dp);
    
}


App::App(string winName, string filelistpath):winName(winName),showIdx(0){
    vector<string> dirlist;
    ifstream f;
    f.open(filelistpath);
    if(!f.is_open()){
        cerr<<filelistpath+" not found!"<<endl;
        exit(1);
    }
    char buffer[256];
    while (!f.eof()) {
        f.getline(buffer, 200);
        dirlist.push_back(buffer);
    }
    
    //training
    CombinedClassifier* classifier = new CombinedClassifier();
    for (int i = 0; i < dirlist.size(); i++) {
        printf("traning data :%d\n",i);
        int64 t0 = getTickCount();
        vector<Mat> _imgs,_mattes;
        loadimage(dirlist[i], _imgs, _mattes);
        classifier->train(_imgs, _mattes);
        printf("traning finished, time cost: %lf", (getTickCount()-t0)/getTickFrequency());
    }
    classifier->exportdata();
}


App::App(string winName, string testpath, string dirname):winName(winName),showIdx(0){
    
    loadimage(testpath+dirname, imgs, mattes);
};

void App::showImg(){
    switch (currentShowState) {
        case SHOW_ORIGIN:
            printf("show origin\n");
            imshow(winName, imgs[showIdx]);
            break;
        case SHOW_WARP:
            printf("show warped\n");
            imshow(winName, warped_imgs[showIdx]);
            break;
        case SHOW_MATTE_GT:
            printf("show origin matte\n");
            imshow(winName, mattes[showIdx]);
            break;
        case SHOW_MATTE_WARP:
            printf("show warped matte\n");
            imshow(winName, warped_mattes[showIdx]);
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
    warped_imgs.resize(imgs.size());
    warped_mattes.resize(mattes.size());
//    for (int i = 0; i < imgs.size(); i++) {
//        tracker->process(imgs[i], optflows[i]);
//    }
    vector<vector<KeyPoint>> img_pts;
    OFFeatureMatcher* matcher = new OFFeatureMatcher(true, imgs, img_pts, mattes);
    vector<vector<DMatch>> matches;
    matches.resize(imgs.size()-1);
    for (int i = 0; i < imgs.size()-1; i++) {
        matcher->registration(i, i+1, warped_imgs[i], warped_mattes[i]);
    }
    warped_imgs[imgs.size()-1] = imgs[imgs.size()-1].clone();
    warped_mattes[mattes.size()-1] = mattes[mattes.size()-1].clone();

    
    printf("calculate finished!\n");
}

void App::changeShowState(){
    printf("currentstate:%d\n",currentShowState);
    currentShowState = (currentShowState+1) % 4;
    showImg();
}

void App::startTraining(){
    
}




void App::testUDC(){
    Mat valid(imgs[0].rows, imgs[0].cols,CV_8UC1);
    valid.setTo(255);
    Mat a,b;
    processUDC(imgs[0], mattes[0], valid, a, b);
}


void App::testLocal(){
    vector<int> a;
    Mat train(15,3,CV_32FC1);
    train.setTo(1.5);
    //LocalClassifier k(Vec3d(12,0,0), train, a);
    Mat b,c;
    processLC(imgs[2], mattes[0], b, c);
}

void App::testGlobal(){
    Mat a,b;
    processGC(imgs[0], mattes[0], a, b);
}

void App::testShape(){
    Mat a,b;
    processSP(imgs[0], mattes[0], a, b);
}

void App::testlearn(){
    //string j = "d";
    CombinedClassifier* g = new CombinedClassifier("wtf");
}



