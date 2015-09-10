//
//  common.cpp
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "common.h"
void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
    ps.clear();
    for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
    kps.clear();
    for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}


#define intrpmnmx(val,min,max) (max==min ? 0.0 : ((val)-min)/(max-min))

void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, const vector<float>& verror, const Scalar& _line_color)
{
    double minVal,maxVal; minMaxIdx(verror,&minVal,&maxVal,0,0,status);
    int line_thickness = 1;
    
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            double alpha = intrpmnmx(verror[i],minVal,maxVal); alpha = 1.0 - alpha;
            Scalar line_color(alpha*_line_color[0],
                              alpha*_line_color[1],
                              alpha*_line_color[2]);
            
            Point p = prevPts[i];
            Point q = nextPts[i];
            
            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);
            
            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );
            
            if (hypotenuse < 1.0)
                continue;
            
            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 1 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 1 * hypotenuse * sin(angle));
            
            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);
            
            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.
            
            p.x = (int) (q.x + 4 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 4 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
            
            p.x = (int) (q.x + 4 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 4 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

double variance(const vector<int>& data){
    double mean = 0;
    for (int i = 0; i < data.size(); i++) {
        mean+=data[i];
    }
    mean = mean / (double)data.size();
    double var = 0;
    for (int i = 0; i < data.size(); i++) {
        var += (mean-data[i])*(mean-data[i]);
    }
    return var/(double)data.size();
}

void getCutout(const Mat& src, const Mat& prob, Mat& cutout){
    Mat mask, tmpprob;
    prob.convertTo(tmpprob, CV_32FC1);
    threshold(tmpprob, mask, 0.1, 1., CV_THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    src.copyTo(cutout, mask);
}

void computeRawDist(const Mat& matte, Mat& raw_dist){
    vector<vector<Point> > contours; vector<Vec4i> hierarchy;
    Mat matte_copy = matte.clone();
    Mat img_copy = matte.clone();
    
    findContours( matte_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    raw_dist.create(matte.size(), CV_32FC1 );
    
    for( int j = 0; j < matte.rows; j++ )
    {
        for( int i = 0; i < matte.cols; i++ )
        {
            raw_dist.at<float>(j,i) = pointPolygonTest( contours[0], Point2f(i,j), true );
        }
    }

}


void drawContour(const Mat& src, const Mat& prob, Mat& dst){
    vector<vector<Point> > contours; vector<Vec4i> hierarchy;
    Mat prob_copy = prob.clone();
    dst = src.clone();
    prob_copy = prob_copy*255;
    prob_copy.convertTo(prob_copy, CV_8UC1);
    findContours( prob_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(dst, contours, 0, Scalar(255,255,0));
//    imshow("d", dst);
//    imshow("p", prob_copy);
//    waitKey(0);
}

void refineProb(Mat& prob){
    for (int i = 0; i < prob.rows; i++) {
        for (int j = 0; j < prob.cols; j++) {
            if (isnan(prob.at<double>(i,j))) {
               // printf("nan!");
                prob.at<double>(i,j)=1;
            }
        }
    }
}
