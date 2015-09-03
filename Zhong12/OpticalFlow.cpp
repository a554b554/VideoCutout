//
//  OpticalFlow.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "OpticalFlow.h"
#include <set>
//////////////////////////////////////////////////////////////////

OFFeatureMatcher::OFFeatureMatcher(
                                   bool _use_gpu,
                                   std::vector<cv::Mat>& imgs_,
                                   std::vector<std::vector<cv::KeyPoint> >& imgpts_,
                                   vector<Mat>& mattes_) :
AbstractFeatureMatcher(_use_gpu),imgpts(imgpts_), imgs(imgs_),mattes(mattes_)
{
    //detect keypoints for all images
    FastFeatureDetector ffd;
    //	DenseFeatureDetector ffd;
    ffd.detect(imgs, imgpts);
}

void OFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches, Mat& output) {
    vector<Point2f> i_pts;
    KeyPointsToPoints(imgpts[idx_i],i_pts);
    
    vector<Point2f> j_pts(i_pts.size());
    
    // making sure images are grayscale
    Mat prevgray,gray;
    if (imgs[idx_i].channels() == 3) {
        cvtColor(imgs[idx_i],prevgray,CV_RGB2GRAY);
        cvtColor(imgs[idx_j],gray,CV_RGB2GRAY);
    } else {
        prevgray = imgs[idx_i];
        gray = imgs[idx_j];
    }
    
    vector<uchar> vstatus(i_pts.size()); vector<float> verror(i_pts.size());
    
#ifdef HAVE_OPENCV_GPU
    if(use_gpu) {
        gpu::GpuMat gpu_prevImg,gpu_nextImg,gpu_prevPts,gpu_nextPts,gpu_status,gpu_error;
        gpu_prevImg.upload(prevgray);
        gpu_nextImg.upload(gray);
        gpu_prevPts.upload(Mat(i_pts).t());
        
        gpu::PyrLKOpticalFlow gpu_of;
        gpu_of.sparse(gpu_prevImg,gpu_nextImg,gpu_prevPts,gpu_nextPts,gpu_status,&gpu_error);
        
        Mat j_pts_mat;
        gpu_nextPts.download(j_pts_mat);
        Mat(j_pts_mat.t()).copyTo(Mat(j_pts));
        
        Mat vstatus_mat,verror_mat;
        gpu_status.download(vstatus_mat);
        gpu_error.download(verror_mat);
        Mat(vstatus_mat.t()).copyTo(Mat(vstatus));
        Mat(verror_mat.t()).copyTo(Mat(verror));
    } else
#endif
    {
      calcOpticalFlowPyrLK(prevgray, gray, i_pts, j_pts, vstatus, verror);
    }
    
    double thresh = 1.0;
    vector<Point2f> to_find;
    vector<int> to_find_back_idx;
    for (unsigned int i=0; i<vstatus.size(); i++) {
        if (vstatus[i] && verror[i] < 12.0) {
            to_find_back_idx.push_back(i);
            to_find.push_back(j_pts[i]);
        } else {
            vstatus[i] = 0;
        }
    }
    
    std::set<int> found_in_imgpts_j;
    Mat to_find_flat = Mat(to_find).reshape(1,to_find.size());
    
    vector<Point2f> j_pts_to_find;
    KeyPointsToPoints(imgpts[idx_j],j_pts_to_find);
    Mat j_pts_flat = Mat(j_pts_to_find).reshape(1,j_pts_to_find.size());
    
    vector<vector<DMatch> > knn_matches;
    //FlannBasedMatcher matcher;
    BFMatcher matcher(CV_L2);
    matcher.radiusMatch(to_find_flat,j_pts_flat,knn_matches,2.0f);
    
    for(int i=0;i<knn_matches.size();i++) {
        DMatch _m;
        if(knn_matches[i].size()==1) {
            _m = knn_matches[i][0];
        } else if(knn_matches[i].size()>1) {
            if(knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
                _m = knn_matches[i][0];
            } else {
                continue; // did not pass ratio test
            }
        } else {
            continue; // no match
        }
        if (found_in_imgpts_j.find(_m.trainIdx) == found_in_imgpts_j.end()) { // prevent duplicates
            _m.queryIdx = to_find_back_idx[_m.queryIdx]; //back to original indexing of points for <i_idx>
            matches->push_back(_m);
            found_in_imgpts_j.insert(_m.trainIdx);
        }
    }
    
    cout << "pruned " << matches->size() << " / " << knn_matches.size() << " matches" << endl;


    // draw flow field
    Mat img_matches;
    imgs[idx_i].copyTo(img_matches);
    //cvtColor(imgs[idx_i],img_matches,CV_GRAY2BGR);
    i_pts.clear(),j_pts.clear();
    for(int i=0;i<matches->size();i++) {
        //if (i%2 != 0) {
        //				continue;
        //			}
        Point i_pt = imgpts[idx_i][(*matches)[i].queryIdx].pt;
        Point j_pt = imgpts[idx_j][(*matches)[i].trainIdx].pt;
        i_pts.push_back(i_pt);
        j_pts.push_back(j_pt);
        vstatus[i] = 1;
    }
    drawArrows(img_matches, i_pts, j_pts, vstatus, verror, Scalar(0,255));
    stringstream ss;
    ss << matches->size() << " matches";
    //		putText(img_matches,ss.str(),Point(10,20),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255),2);
    ss.clear(); ss << "flow_field";
//    imshow( ss.str(), img_matches );
//    waitKey(0);
    output = img_matches.clone();
    
   // destroyWindow(ss.str());

}

void OFFeatureMatcher::registration(int idx_i, int idx_j, Mat &registrated_img, Mat& registrated_matte){
    vector<DMatch> matches;
    Mat tmp;
    MatchFeatures(idx_i, idx_j, &matches, tmp);
    //warp image.
    vector<Point2f> i_pts,j_pts;
    for (int i = 0; i < matches.size(); i++) {
        Point i_pt = imgpts[idx_i][matches[i].queryIdx].pt;
        Point j_pt = imgpts[idx_j][matches[i].trainIdx].pt;
        i_pts.push_back(i_pt);
        j_pts.push_back(j_pt);
        if (mattes[idx_i].at<uchar>(i_pt)!=0) {
            continue;
        }
    }
    Mat M = findHomography(i_pts, j_pts, CV_RANSAC);
    Mat warped,warped_mat;
    warpPerspective(imgs[idx_i], warped, M, imgs[0].size());
    
    //warp matte
    i_pts.clear(); j_pts.clear();
    for (int i = 0; i < matches.size(); i++) {
        Point i_pt = imgpts[idx_i][matches[i].queryIdx].pt;
        Point j_pt = imgpts[idx_j][matches[i].trainIdx].pt;
        if (mattes[idx_i].at<uchar>(i_pt)==0) {
            continue;
        }
        //circle(mattes[idx_i], i_pt, 2, CV_RGB(255, 0, 0));
        i_pts.push_back(i_pt);
        j_pts.push_back(j_pt);
    }
//    imshow("circle", mattes[idx_i]);
//    waitKey(0);
    Mat M_matte = findHomography(i_pts, j_pts, CV_RANSAC);
    warpPerspective(mattes[idx_i], warped_mat, M_matte, mattes[0].size());
    
    Mat errormat(imgs[idx_i].size(),CV_64FC1);
    Mat minus = imgs[idx_j]-warped;
    for (int i = 0; i < imgs[0].rows; i++) {
        for (int j = 0; j < imgs[0].cols; j++) {
            Vec3d color = minus.at<Vec3b>(i,j);
            double val = color[0]*color[0]+color[1]*color[1]+color[2]*color[2];
            if (val == 0) {
                errormat.at<double>(i,j) = val;
            }
            else{
                errormat.at<double>(i,j) = 1;
            }
            
        }
    }
    
    //debug
//    imshow("err", errormat);
//    
//    imshow("after", warped_mat);
//    imshow("before", mattes[idx_i]);
//    imshow("align", mattes[idx_j]);
//    imshow("align + after", 0.5*mattes[idx_j]+0.5*warped_mat);
//    imshow("align + before", 0.5*mattes[idx_j]+0.5*mattes[idx_i]);
//    
//    imshow("align + after img", 0.5*imgs[idx_j]+0.5*warped);
//    imshow("align + before img", 0.5*imgs[idx_j]+0.5*imgs[idx_i]);
//    waitKey(0);
    warped.copyTo(registrated_img);
    warped_mat.copyTo(registrated_matte);
    
    
}



