//
//  common.h
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __Zhong12__common__
#define __Zhong12__common__
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <list>
#include <set>
#include <stdio.h>
#include <string>
#include <fstream>
#include "Eigen/Sparse"

using namespace std;
using namespace cv;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Td;

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);
void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);

void drawArrows(cv::Mat& frame, const std::vector<cv::Point2f>& prevPts, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status, const std::vector<float>& verror, const cv::Scalar& line_color = cv::Scalar(0, 0, 255));

double variance(const vector<int>& data);

void getCutout(const Mat& src, const Mat& prob, double low, Mat& cutout);
void getCutout2(const Mat& src, const Mat& prob, Mat& cutout);

void computeRawDist(const Mat& matte, Mat& raw_dist, double minArea = 30);

void drawContour(const Mat& src, const Mat& prob, Mat& dst);

void refineProb(Mat& prob);

void getBinaryProbabilityMap(const Mat& prob, Mat& binary, double low, double high);

void minmaxFilter(const Mat& src, Mat& dst, int flag);//flag = 0:=minfilter

void combinedConfidenceMap(const Mat& prob, const Mat& conf, Mat& dst);

void solveMatte(const Mat& src, const Mat& constmap, const Mat& constval, const Mat& prob, const Mat& conf, Mat& dst);

//term is defined in the formula 12.
void getL(const Mat& src, const Mat& trimap, SpMat& laplacian);

void loadSPmat(string filename, SpMat& sp);
void compareMat(SpMat& mat1, SpMat& mat2);

#endif /* defined(__Zhong12__common__) */
