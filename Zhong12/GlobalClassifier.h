//
//  GlobalClassifier.h
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__GlobalClassifier__
#define __Zhong12__GlobalClassifier__

#include <stdio.h>
#include "common.h"
#include "UDC.h"

// matte may contains user input
void processGC(const Mat& img, const Mat& matte, const Mat& raw_dist, Mat& probmat, Mat& confmat);



#endif /* defined(__Zhong12__GlobalClassifier__) */
