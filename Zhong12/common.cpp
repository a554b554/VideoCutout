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

void getCutout(const Mat& src, const Mat& prob, double low, Mat& cutout){
    Mat mask, tmpprob;
    prob.convertTo(tmpprob, CV_32FC1);
    threshold(tmpprob, mask, low, 1., CV_THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
   // cout<<mask;
    //imshow("mask", mask);
    //debug
//    Mat show = src.clone();
//    vector<vector<Point> > contours; vector<Vec4i> hierarchy;
//    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
//    
//    for( int i = 0; i< contours.size(); i++ )
//    {
//        Scalar color = Scalar(255,255,0);
//        drawContours(show, contours, i, color, 2, 8, hierarchy, 0, Point());
//        imshow("show", show);
//        printf("%d",i);
//        waitKey(0);
//    }
    
    
    src.copyTo(cutout, mask);
}


void getBinaryProbabilityMap(const Mat& prob, Mat& binary, double low, double high){
    binary = prob.clone();
    binary = binary*255;
    binary.convertTo(binary, CV_8UC1);
    threshold(binary, binary, low, high, CV_THRESH_BINARY);
//    imshow("binary", binary);
//    waitKey(0);
}

void computeRawDist(const Mat& matte, Mat& raw_dist, double minArea){
    vector<vector<Point> > contours; vector<Vec4i> hierarchy;
    Mat matte_copy = matte.clone();
    Mat img_copy = matte.clone();
    
    //hierarchy: [Next, Previous, Child, Parent]
    
    findContours( matte_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    //debug show
    cvtColor(matte_copy, matte_copy, CV_GRAY2BGR);
    vector<int> good;
    for (int i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) < minArea) {
            good.push_back(0);
            continue;
        }
        good.push_back(1);
        drawContours(matte_copy, contours, i, Scalar(255,255,0));

    }
//    imshow("contour", matte_copy);
//    waitKey(0);
    
    
    raw_dist.create(matte.size(), CV_64FC1 );
    
    for( int j = 0; j < matte.rows; j++ )
    {
        for( int i = 0; i < matte.cols; i++ )
        {
            double dist = pointPolygonTest(contours[0], Point2f(i,j), true);
            int cid = 0;
            for (;cid != -1;cid = hierarchy[cid][0]) {
                
                //show current progress
//                Mat current(matte.size(),CV_8UC3);
//                current.setTo(0);
//                circle(current, Point2f(i,j), 1, Scalar(0,0,255));
//                drawContours(current, contours, cid, Scalar(255,255,0));
//                imshow("progress", current);
//                waitKey(0);
                
                
                
                
                if (good[cid] == 0) { // if is not good, do next iteration
                    continue;
                }
                double tmp = pointPolygonTest(contours[cid], Point2f(i,j), true);
                if (tmp <= 0) { // if point is outside a contour
                    dist = max(tmp, dist);
                }
                else{// if point is inside a contour
                    dist = tmp;
                    int kid = hierarchy[cid][2];//child contour
                    for (;kid != -1;kid = hierarchy[kid][0]) {
                        
                        if (good[kid] == 0) {// bad contour
                            continue;
                        }

                        
                        double tmp2 = pointPolygonTest(contours[kid], Point2f(i,j), true);
                        if (tmp2 >= 0) { // inside the child contour
                            dist = -tmp2;//which means out of a contour
                            break;
                        }
                        else{
                            dist = min(-tmp2,dist);
                        }
                    }
                    break;
                }
            }
            raw_dist.at<double>(j,i) = dist;
            
        }
    }

    
    //debug show
//    double minVal; double maxVal;
//    minMaxLoc( raw_dist, &minVal, &maxVal, 0, 0, Mat() );
//    minVal = abs(minVal); maxVal = abs(maxVal);
//    Mat drawing = Mat::zeros( matte.size(), CV_8UC3 );
//    
//    for( int j = 0; j < matte.rows; j++ )
//    {
//        for( int i = 0; i < matte.cols; i++ )
//        {
//            
//            if( raw_dist.at<double>(j,i) < -5 ){
//                drawing.at<Vec3b>(j,i)[0] = 255;//- (int) abs(raw_dist.at<float>(j,i))*255/minVal;
//            }
//            else if( raw_dist.at<double>(j,i) > 5 )
//            {
//
//                drawing.at<Vec3b>(j,i)[2] = 255; //- (int) raw_dist.at<float>(j,i)*255/maxVal;
//            }
//            else
//            {
//                drawing.at<Vec3b>(j,i)[0] = 255; drawing.at<Vec3b>(j,i)[1] = 255; drawing.at<Vec3b>(j,i)[2] = 255;
//            }
//        }
//    }
//    imshow("draw", drawing);
//    waitKey(0);
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
    Mat binary,dist;
    getBinaryProbabilityMap(prob, binary, 100, 255);
    imshow("binaryprob", binary);
    
    computeRawDist(binary, dist,300);
    for (int i = 0; i < prob.rows; i++) {
        for (int j = 0; j < prob.cols; j++) {
            if (dist.at<double>(i,j) > 0) {
                prob.at<double>(i,j) = 1;
            }
        }
    }
}

//flag=0 := minfilter
//flag=1 := maxfilter
const int winSize = 3;
void minmaxFilter(const Mat& src, Mat& dst, int flag){
    dst = src.clone();
    for (int i = winSize; i < src.rows-winSize-1; i++) {
        for (int j = winSize; j < src.cols-winSize-1; j++) {
            int length = 2*winSize-1;
            Rect roi(j,i,length,length);
            double minval,maxval;
            minMaxIdx(src(roi), &minval, &maxval);
            if (flag==0) { //use minfilter
                dst.at<double>(i,j) = minval;
            }
            else if (flag==1) {
                dst.at<double>(i,j) = maxval;
            }
        }
    }
}



void combinedConfidenceMap(const Mat& prob, const Mat& conf, Mat& dst){
    Mat minp,maxp;
    minmaxFilter(prob, minp, 0);
    minmaxFilter(prob, maxp, 1);
    dst = conf.clone();
    for (int i = 0; i < prob.rows; i++) {
        for (int j = 0; j < prob.cols; j++) {
            dst.at<double>(i,j) = (1-fabs(minp.at<double>(i,j)-
                                          maxp.at<double>(i,j)))*
            fabs(prob.at<double>(i,j)-0.5)*conf.at<double>(i,j)*2.0;
        }
    }
}

//unknow = 0 := known 1:= unknow region
void binary2pt(const Mat& src, vector<Point2f>& pts){
    for (int i = 1; i < src.rows-1; i++) {
        for (int j = 1; j < src.cols-1; j++) {
            if (src.at<double>(i,j) == 1) {
                pts.push_back(Point2f(j,i));
            }
        }
    }
}

void solveMatte(const Mat& src, const vector<Point2f>& unknow_pts, Mat& dst){
    int64 t0 = getTickCount();
    cout<<"solve matte cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
}

void getL(const Mat& src, const vector<Point2f>& unknow_pts, Mat& laplacian){
    
}















