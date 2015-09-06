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

int main(int argc, const char * argv[]){
    
    string testPath = "../../Zhong12-SIGA-dataset/TEST/";
    string dirname = argv[1];
    //testPath = testPath + argv[1] + "/";
    App app("app", testPath, dirname);
    app.testUDC();
}





