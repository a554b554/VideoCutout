//
//  GMM.h
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__GMM__
#define __Zhong12__GMM__

#include <stdio.h>
#include "common.h"
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

class GMM
{
public:
    static const int componentsCount = 3;
    
    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;
    
    void initLearning();
    void addSample( int ci, const Vec3d color );
    void learning(const vector<Vec3d>& colors);
    void endLearning();
    void updateModel();
    
private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;
    
    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];
    
    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};


#endif /* defined(__Zhong12__GMM__) */
