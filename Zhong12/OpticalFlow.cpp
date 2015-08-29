//
//  OpticalFlow.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "OpticalFlow.h"
#include <set>


void FeatureTracker::process(Mat &frame,Mat &output){
    //得到灰度图
    cvtColor (frame,gray,CV_BGR2GRAY);
    frame.copyTo (output);
    //特征点太少了，重新检测特征点
    if(addNewPoint()){
        detectFeaturePoint ();
        //插入检测到的特征点
        points[0].insert (points[0].end (),features.begin (),features.end ());
        initial.insert (initial.end (),features.begin (),features.end ());
    }
    //第一帧
    if(gray_prev.empty ()){
        gray.copyTo (gray_prev);
    }
    //根据前后两帧灰度图估计前一帧特征点在当前帧的位置
    //默认窗口是15*15
    
    calcOpticalFlowPyrLK (
                          gray_prev,//前一帧灰度图
                          gray,//当前帧灰度图
                          points[0],//前一帧特征点位置
                          points[1],//当前帧特征点位置
                          status,//特征点被成功跟踪的标志
                          err);//前一帧特征点点小区域和当前特征点小区域间的差，根据差的大小可删除那些运动变化剧烈的点
    int k = 0;
    //去除那些未移动的特征点
    for(int i=0;i<points[1].size ();i++){
        if(acceptTrackedPoint (i)){
            initial[k]=initial[i];
            points[1][k++] = points[1][i];
        }
    }
    points[1].resize (k);
    initial.resize (k);
    //标记被跟踪的特征点
    drawTrackedPoint (frame,output);
    //为下一帧跟踪初始化特征点集和灰度图像
    std::swap(points[1],points[0]);
    cv::swap(gray_prev,gray);
}

void FeatureTracker::detectFeaturePoint(){
    goodFeaturesToTrack (gray,//输入图片
                         features,//输出特征点
                         max_count,//特征点最大数目
                         qlevel,//质量指标
                         minDist);//最小容忍距离
}

bool FeatureTracker::addNewPoint(){
    return points[0].size() <= 10;
}

bool FeatureTracker::acceptTrackedPoint(int i){
    return status[i]&&
    (abs(points[0][i].x-points[1][i].x)+
     abs(points[0][i].y-points[1][i].y) >2);
}

void FeatureTracker::drawTrackedPoint(Mat &frame,Mat &output){
    for(int i=0;i<points[i].size ();i++){
        //当前特征点到初始位置用直线表示
        line(output,initial[i],points[1][i],Scalar::all (0));
        //当前位置用圈标出
        circle(output,points[1][i],3,Scalar::all(0),(-1));
    }
}


//////////////////////////////////////////////////////////////////

OFFeatureMatcher::OFFeatureMatcher(
                                   bool _use_gpu,
                                   std::vector<cv::Mat>& imgs_,
                                   std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
AbstractFeatureMatcher(_use_gpu),imgpts(imgpts_), imgs(imgs_)
{
    //detect keypoints for all images
    FastFeatureDetector ffd;
    //	DenseFeatureDetector ffd;
    ffd.detect(imgs, imgpts);
}

void OFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {
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
    imshow( ss.str(), img_matches );
    waitKey(0);
    
   // destroyWindow(ss.str());

}




