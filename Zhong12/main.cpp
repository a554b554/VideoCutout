//
//  main.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "App.h"
#include "GMM.h"
#include <string>
#include "Eigen/Sparse"


using namespace std;
using namespace cv;

static void help(){
    cout<<"usage:\n"
    <<"[test data name]"<<endl;
}


int main1(int argc, const char * argv[]) {
    // insert code here...
    string testPath = "../../Zhong12-SIGA-dataset/TEST/";
    string dirname = argv[1];
    //testPath = testPath + argv[1] + "/";
    App app("app", testPath, dirname);
    app.calcOpticalFlows();
    app.testUDC();
    app.showImg();
    while (1) {
        int c = cvWaitKey(0);
        switch (c) {
            case 63232: //up
                break;
            case 63233: //down
                break;
            case 63234: //left
                app.prevImg();
                break;
            case 63235: //right
                app.nextImg();
                break;
            case 'a':
                app.changeShowState();
                break;
                
            default:
                break;
        }
    }
    return 0;
}

int maintest(int argc, const char * argv[]){
    
    vector<Vec3d> aa;
    for (int i = 0; i < 40000; i++) {
        Vec3d a(rand()%300+100,rand()%400,0);
        aa.push_back(a);
    }
    Mat af;
    GMM g(af);
    g.learning(aa);
    
    return 0;
}


int main(int argc, const char * argv[]){

    
    string testPath = "../../Zhong12-SIGA-dataset/TEST/";
    string dirname = "DEBUG2";
//    testPath = testPath + argv[1] + "/";
    App* app = new App("app", testPath, dirname);
  
    
    
    vector<string> list;
    parse("../../config/datalist.cfg", list);
    //app->reverse();
    //app->calcOpticalFlows();
    app->start2(list);
    
    //App* app = new App("app", "./filelist.txt");
    //app.testUDC();
    //app.calcOpticalFlows();
    return 0;
}

int maincut(int argc, const char * argv[]){
    if (argc != 3) {
        cout<<"./cutout inputpath outputpath"<<endl;
        return 1;
    }
    string dir = argv[1];
    string outputpath = argv[2];
    vector<Mat> imgs, mattes;
    loadimage(dir, imgs, mattes);
    for (int i = 0; i < imgs.size(); i++) {
        Mat out;
        getCutout2(imgs[i], mattes[i], out);
        imwrite(outputpath+"/"+to_string(i)+".jpg", out);
    }
    return 0;
}



int main231(int argc, const char * argv[]){
    App* app = new App("app", "../../config/train.cfg");
    return 0;
}

int mainvvv(int argc, const char* argv[])
{
    /*Mat_<double> samples = (Mat_<double>(3, 3) << 1.0, 2.0, 3.0,
     4.0, 5.0, 6.0,
     7.0, 8.0, 9.0);*/
    Mat samples;
    

    samples = imread("../../Zhong12-SIGA-dataset/TEST/DEBUG/001.jpg");
    Mat trimap(samples.size(),CV_8UC1);
    trimap.setTo(1);
    SpMat bb(samples.cols*samples.rows,samples.cols*samples.rows);
    //SpMat b(4,4);
    getL(samples, trimap, bb);
    
    return 0;
}


//this is test for matting.
int main2333(int argc, const char * argv[]){
    
    if (argc!=4) {
        cout<<"usage: file1 filealpha outputname"<<endl;
        exit(0);
    }
    
    string file1 = argv[1];
    string filealpha = argv[2];
    string output = argv[3];
    string base = "../../matting/";
    string file = "kid1";
    Mat img = imread(file1);
    Mat scribs = imread(filealpha);
    
    Mat png[3];
    split(img, png);
    
    img.convertTo(img, CV_32F);
    img = img/255;
    scribs.convertTo(scribs, CV_32F);
    scribs = scribs/255;

    //img.create(50, 100, CV_32FC3);
    //scribs.create(50, 100, CV_32FC3);
    
    
    Mat constval = abs(img-scribs);
    cvtColor(constval, constval, CV_BGR2GRAY);
    cvtColor(scribs, scribs, CV_BGR2GRAY);
    Mat constmap;
    threshold(constval, constmap, 0.1, 1, CV_THRESH_BINARY);
    
    SpMat laplacian(img.rows*img.cols,img.rows*img.cols);
    
    constval = scribs.mul(constmap);
    
    
    getL(img, constmap, laplacian);
    SpMat D(laplacian.rows(), laplacian.cols());

    vector<Td> vals;
    int len = laplacian.rows();
    int count = -1;
    for (int i = 0; i < constmap.cols; i++) {
        for (int j = 0; j < constmap.rows; j++) {
            count++;
            if (constmap.at<float>(j,i)==0) {
                continue;
            }
            Td t(count,count,1);
            
            vals.push_back(t);
        }
    }
    D.setFromTriplets(vals.begin(), vals.end());

    double lambda = 100;
    Eigen::VectorXd b(len);
    
    count = 0;
    for (int i = 0; i < constmap.cols; i++) {
        for (int j = 0; j < constmap.rows; j++) {
            b[count] = (double)constmap.at<float>(j,i)*lambda*constval.at<float>(j,i);
            count++;
        }
    }
    
    
    SpMat A = laplacian+lambda*D;
    
    
    Eigen::SimplicialCholesky<SpMat> sol;  // performs a Cholesky factorization of A

    int64 t = getTickCount();
    sol.compute(A);
    cout<<sol.info()<<endl;
    Eigen::VectorXd x = sol.solve(b);
    cout<<"solving cost: "<<(getTickCount()-t)/getTickFrequency()<<endl;
    
    
//    cout<<x<<endl;

  
    
    
    Mat alpha(img.size(),CV_32F);
    count = 0;
    for (int i = 0; i < alpha.cols; i++) {
        for (int j = 0; j < alpha.rows; j++) {
            if(x[count]<0){
                x[count]=0;
            }
            if (x[count]>1) {
                x[count]=1;
            }
            alpha.at<float>(j,i) = x[count];
            count++;
        }
    }
    
    
    imshow("alp", alpha);
    alpha = alpha*255;
    alpha.convertTo(alpha, CV_8UC1);
    imwrite(output, alpha);
    
    vector<Mat> channel;
    channel.push_back(png[0]);
    channel.push_back(png[1]);
    channel.push_back(png[2]);
    channel.push_back(alpha);
    Mat dst;
    merge(channel, dst);
    imwrite("_"+output, dst);
    waitKey(0);
    
    return 0;
}


