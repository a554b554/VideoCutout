//
//  main.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "App.h"
#include "GMM.h"
#include <string>
#include "Eigen/Sparse"


using namespace std;
using namespace cv;

static void help(){
    cout<<"usage:\n"
    <<"[test data name]"<<endl;
}


int main1(int argc, const char * argv[]) {
    // insert code here...
    string testPath = "../../Zhong12-SIGA-dataset/TEST/";
    string dirname = argv[1];
    //testPath = testPath + argv[1] + "/";
    App app("app", testPath, dirname);
    app.calcOpticalFlows();
    app.testUDC();
    app.showImg();
    while (1) {
        int c = cvWaitKey(0);
        switch (c) {
            case 63232: //up
                break;
            case 63233: //down
                break;
            case 63234: //left
                app.prevImg();
                break;
            case 63235: //right
                app.nextImg();
                break;
            case 'a':
                app.changeShowState();
                break;
                
            default:
                break;
        }
    }
    return 0;
}

int maintest(int argc, const char * argv[]){
    
    vector<Vec3d> aa;
    for (int i = 0; i < 40000; i++) {
        Vec3d a(rand()%300+100,rand()%400,0);
        aa.push_back(a);
    }
    Mat af;
    GMM g(af);
    g.learning(aa);
    
    return 0;
}


int mainq(int argc, const char * argv[]){

    
    string testPath = "../../Zhong12-SIGA-dataset/TEST/";
    string dirname = "DEBUG";
//    testPath = testPath + argv[1] + "/";
    App* app = new App("app", testPath, dirname);
  
    
    app->calcOpticalFlows();
    vector<string> list;
    parse("../../config/datalist.cfg", list);
    
    app->start(list);
    
    //App* app = new App("app", "./filelist.txt");
    //app.testUDC();
    //app.calcOpticalFlows();
    return 0;
}

int main231(int argc, const char * argv[]){
    App* app = new App("app", "../../config/train.cfg");
    return 0;
}

int main(int argc, const char* argv[])
{
    /*Mat_<double> samples = (Mat_<double>(3, 3) << 1.0, 2.0, 3.0,
     4.0, 5.0, 6.0,
     7.0, 8.0, 9.0);*/
    Mat samples;
    

    samples = imread("../../Zhong12-SIGA-dataset/TEST/DEBUG/001.jpg");
    Mat trimap(samples.size(),CV_8UC1);
    trimap.setTo(1);
    SpMat bb(samples.cols*samples.rows,samples.cols*samples.rows);
    //SpMat b(4,4);
    getL(samples, trimap, bb);
    
    return 0;
}



