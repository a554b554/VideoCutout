//
//  ShapePrior.h
//  Zhong12
//
//  Created by DarkTango on 9/1/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__ShapePrior__
#define __Zhong12__ShapePrior__

#include <stdio.h>
#include "common.h"
static const double sigmas2 = 25;
void processSP(const Mat& img, const Mat& matte, Mat& probmat, Mat& confmat);


#endif /* defined(__Zhong12__ShapePrior__) */
