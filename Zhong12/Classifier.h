//
//  Classifier.h
//  Zhong12
//
//  Created by DarkTango on 8/30/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__Classifier__
#define __Zhong12__Classifier__
#include "common.h"
#include <stdio.h>
class Classifier{
    virtual double probability(const Vec3d color) = 0;
    virtual double confidence(const Vec3d color) = 0;
};

#endif /* defined(__Zhong12__Classifier__) */
