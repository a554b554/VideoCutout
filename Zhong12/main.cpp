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


using namespace std;
using namespace cv;

static void help(){
    cout<<"usage:\n"
    <<"[test data name]"<<endl;
}


int main(int argc, const char * argv[]) {
    // insert code here...
    string testPath = "../../Zhong12-SIGA-dataset/TEST/";
    testPath = testPath + argv[1] + "/";
    App app("app", testPath);
    app.showImg();
    while (1) {
        int c = cvWaitKey(0);
        switch (c) {
            case 63232: //up
                printf("up\n");
                break;
            case 63233: //down
                printf("down\n");
                break;
            case 63234: //left
                printf("left\n");
                app.prevImg();
                break;
            case 63235: //right
                printf("right\n");
                app.nextImg();
                break;
            
                
            default:
                break;
        }
    }
}
