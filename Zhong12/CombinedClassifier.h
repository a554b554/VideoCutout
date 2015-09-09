//
//  CombinedClassifier.h
//  Zhong12
//
//  Created by DarkTango on 9/2/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__CombinedClassifier__
#define __Zhong12__CombinedClassifier__

#include <stdio.h>
#include "common.h"
#include "UDC.h"
#include "LocalClassifier.h"
#include "GlobalClassifier.h"
#include "ShapePrior.h"
#include <fstream>
#include "RegistrationError.h"

struct featureVector{
    double ru;
    double rl;
    double rg;
    double rs;
    double e;
    double dist2(const featureVector& vec)const;
    void print();
    featureVector operator*(double num);
};


class CombinedClassifier{
public:
    static constexpr double sigmad2 = 0.01;
    static const int cSize = 20;
    static constexpr long interval = cSize*cSize*cSize*cSize*cSize;
    static featureVector getCorByID(long i);
    static int getNearestVectorID(featureVector v);
    CombinedClassifier();
    CombinedClassifier(const string filepath);//for loading learned data
    void init();
    void train(const vector<Mat>& imgs, const vector<Mat>& mattes_gt, const vector<Mat>& remats, const vector<Mat>& mattes_warped);
    void addSample(featureVector v, bool addtoForeground);
    double prob(featureVector v);
    double conf(featureVector v);
    void exportdata();
    
private:
    double fLattice[interval]; //store the foreground trained data.
    double bLattice[interval]; //store the background trained data.
    
};





#endif /* defined(__Zhong12__CombinedClassifier__) */
