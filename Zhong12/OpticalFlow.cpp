//
//  OpticalFlow.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "OpticalFlow.h"

class FrameProcessor{
public:
    virtual void process(Mat &input,Mat &ouput)=0;
};

//特征跟踪类，继承自帧处理基类
class FeatureTracker :  public FrameProcessor{
    Mat gray;  //当前灰度图
    Mat gray_prev;  //之前的灰度图
    vector<Point2f> points[2];//前后两帧的特征点
    vector<Point2f> initial;//初始特征点
    vector<Point2f> features;//检测到的特征
    int max_count; //要跟踪特征的最大数目
    double qlevel; //特征检测的指标
    double minDist;//特征点之间最小容忍距离
    vector<uchar> status; //特征点被成功跟踪的标志
    vector<float> err; //跟踪时的特征点小区域误差和
public:
    FeatureTracker():max_count(500),qlevel(0.01),minDist(10.){}
    void process(Mat &frame,Mat &output){
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
        handleTrackedPoint (frame,output);
        //为下一帧跟踪初始化特征点集和灰度图像
        std::swap(points[1],points[0]);
        cv::swap(gray_prev,gray);
    }
    
    void detectFeaturePoint(){
        goodFeaturesToTrack (gray,//输入图片
                             features,//输出特征点
                             max_count,//特征点最大数目
                             qlevel,//质量指标
                             minDist);//最小容忍距离
    }
    bool addNewPoint(){
        //若特征点数目少于10，则决定添加特征点
        return points[0].size ()<=10;
    }
    
    //若特征点在前后两帧移动了，则认为该点是目标点，且可被跟踪
    bool acceptTrackedPoint(int i){
        return status[i]&&
        (abs(points[0][i].x-points[1][i].x)+
         abs(points[0][i].y-points[1][i].y) >2);
    }
    
    //画特征点
    void  handleTrackedPoint(Mat &frame,Mat &output){
        for(int i=0;i<points[i].size ();i++){
            //当前特征点到初始位置用直线表示
            line(output,initial[i],points[1][i],Scalar::all (0));
            //当前位置用圈标出
            circle(output,points[1][i],3,Scalar::all(0),(-1));
        }
    }
};