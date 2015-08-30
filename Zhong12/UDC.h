//
//  UDC.h
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__UDC__
#define __Zhong12__UDC__

#include <stdio.h>
#include "common.h"
#include "GMM.h"
#include "Classifier.h"
using namespace std;
using namespace cv;

static void processUDC(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat){
    
}


class UDC : public Classifier{
private:
    GMM* fGMM;
    GMM* bGMM;
public:
    UDC(const vector<Vec3d>& fgdSamples, const vector<Vec3d>& bdgSamples);
};


#endif /* defined(__Zhong12__UDC__) */
