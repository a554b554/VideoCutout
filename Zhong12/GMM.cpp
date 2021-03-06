//
//  GMM.cpp
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "GMM.h"
GMM::GMM( Mat& _model )
{
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );
    
    model = _model.clone();
    
    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;
    
    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
            calcInverseCovAndDeterm( ci );
}

double GMM::operator()( const Vec3d& color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d& color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
        + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
        + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3d& color ) const
{
    int k = 0;
    double max = 0;
    
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample( int ci, const Vec3d& color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::learning(const vector<Vec3d>& colors){
    CV_Assert(!colors.empty());
    Mat samples(colors);
     samples.convertTo(samples, CV_32FC1);
    Mat lab;
    kmeans(samples, componentsCount, lab, TermCriteria( CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
    initLearning();
    for (int i = 0; i < colors.size(); i++) {
        this->addSample(lab.at<int>(i,0), colors[i]);
    }
    endLearning();
    vector<int> comp;
    for (int i = 0; i < 50; i++) { //max iter num
        comp.clear();
        Mat oldmodel = model.clone();
        for (int j = 0; j < colors.size(); j++) {
            comp.push_back(whichComponent(colors[j]));
        }
        
        //test draw
//        Mat demo(400,400,CV_8UC3);
//        demo.setTo(Scalar::all(0));
//        for (int j = 0; j < colors.size(); j++) {
//            Point p(colors[j][0],colors[j][1]);
//            if (comp[j]==0) {
//                circle(demo, p, 2, CV_RGB(255, 0, 0));
//            }
//            else if (comp[j]==1){
//                circle(demo, p, 2, CV_RGB(0, 255, 0));
//            }
//            else{
//                circle(demo, p, 2, CV_RGB(0, 0, 255));
//            }
//        }
//        cout<<"iter: "<<i<<endl;
        
        updateModel();
        initLearning();
        for (int j = 0; j < colors.size(); j++) {
            this->addSample(comp[j], colors[j]);
        }
        endLearning();
        double adiff = norm(oldmodel, model);
        if (adiff < 1) { //iter threshold.
            //printf("terminate by threshold.\n");
            break;
        }
        if (i == 49) {
            //printf("terminate by max iter.\n");
        }
    }
    //compute error and correct.
    for (int i = 0; i < colors.size(); i++) {
        for (int j = 0; j < componentsCount; j++) {
            error[j] += (*this)(j,colors[i]) * (1-(*this)(colors[i]));
            correct[j] += (*this)(j,colors[i]) * (*this)(colors[i]);
        }
//        error[comp[i]]+=(*this)(comp[i],colors[i]) * (1-(*this)(colors[i]));
//        correct[comp[i]]+=(*this)(comp[i],colors[i]) * (*this)(colors[i]);
    }
}


void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n/totalSampleCount;
            
            double* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;
            
            double* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];
            
            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }
            
            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm =
        covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
        
        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}

void GMM::updateModel(){
//    model = _model;
//    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
//    coefs = model.ptr<double>(0);
//    mean = coefs + componentsCount;
//    cov = mean + 3*componentsCount;
    for (int i = 0; i < componentsCount; i++) {
        model.at<double>(0,i) = coefs[i];
    }
    for (int i = 0; i < 3*componentsCount; i++) {
        model.at<double>(0,i+componentsCount) = mean[i];
    }
    for (int i = 0; i < 9*componentsCount; i++) {
        model.at<double>(0,i+componentsCount+3*componentsCount) = cov[i];
    }
}

double GMM::quantity(const Vec3d& color, bool isForeground){
    double ans = 0;
    for (int i = 0; i < componentsCount; i++) {
        double q = correct[i]/(correct[i]+error[i]);
        if (!isForeground) {
            q = 1 - q;
        }
        ans += (*this)(i, color) * q;
    }
    return ans;
}
