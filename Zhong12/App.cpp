//
//  App.cpp
//  Zhong12
//
//  Created by DarkTango on 8/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "App.h"


void loadimage(string dirname, vector<Mat>& imgs, vector<Mat>& mattes, int code){
    string path = dirname + "/";
    string alpha = dirname + "_alpha/";
    
    
    //load image
    DIR *dp;
    struct dirent *dirp;
    if((dp=opendir(path.c_str()))==NULL){
        perror("opendir error");
        free(dp);
        exit(1);
    }
    
    struct stat buf;
    while((dirp=readdir(dp))!=NULL){
        if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
            continue;
        string fname = path+dirp->d_name;
        Mat img = imread(fname);
        if (img.empty()) {
            continue;
        }
        if (code != MAGIC_NUMBER_BGR) {
            cvtColor(img, img, code);
        }
        imgs.push_back(img);
    }
    closedir(dp);
    
    //load mattes.
    if((dp=opendir(alpha.c_str()))==NULL){
        perror("opendir error");
        free(dp);
        exit(1);
    }
    
    while((dirp=readdir(dp))!=NULL){
        if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
            continue;
        string fname = alpha+dirp->d_name;
        mattes.push_back(imread(fname,0));
        //        imshow("matte", mattes[mattes.size()-1]);
        //        waitKey(0);
    }
    closedir(dp);
    
}

void parse(string filepath, vector<string>& content){
    ifstream f;
    f.open(filepath);
    if(!f.is_open()){
        cerr<<filepath+" not found!"<<endl;
        exit(1);
    }
    char buffer[256];
    while (!f.eof()) {
        f.getline(buffer, 200);
        content.push_back(buffer);
    }
}


App::App(string winName, string filelistpath):winName(winName),showIdx(0){
    vector<string> dirlist;
    parse(filelistpath, dirlist);
    
    //training
    CombinedClassifier* classifier = new CombinedClassifier();
    for (int i = 0; i < dirlist.size(); i++) {
        printf("traning data :%d\n",i);
        int64 t0 = getTickCount();
        //vector<Mat> _imgs,_mattes;
        loadimage(dirlist[i], imgs, mattes);
        calcOpticalFlows();
        classifier->train(imgs, mattes, remats, warped_mattes);
        printf("traning finished, time cost: %lf", (getTickCount()-t0)/getTickFrequency());
        clear();
        
    }
    classifier->exportdata();
}


App::App(string winName, string testpath, string dirname):winName(winName),showIdx(0){
    
    loadimage(testpath+dirname, imgs, mattes);
};

void App::showImg(){
    switch (currentShowState) {
        case SHOW_ORIGIN:
            printf("show origin\n");
            imshow(winName, imgs[showIdx]);
            break;
        case SHOW_WARP:
            printf("show warped\n");
            imshow(winName, warped_imgs[showIdx]);
            break;
        case SHOW_MATTE_GT:
            printf("show origin matte\n");
            imshow(winName, mattes[showIdx]);
            break;
        case SHOW_MATTE_WARP:
            printf("show warped matte\n");
            imshow(winName, warped_mattes[showIdx]);
            break;
        default:
            break;
    }
}

void App::nextImg(){
    showIdx = (showIdx+1) % imgs.size();
    showImg();
}

void App::prevImg(){
    showIdx = (showIdx - 1)>=0? (showIdx-1):(imgs.size()-1);
    showImg();
}

void App::calcOpticalFlows(){
   // printf("calculating optical flow....\n");
    int64 t0 = getTickCount();
    warped_imgs.resize(imgs.size());
    warped_mattes.resize(mattes.size());
//    for (int i = 0; i < imgs.size(); i++) {
//        tracker->process(imgs[i], optflows[i]);
//    }
    vector<vector<KeyPoint>> img_pts;
    OFFeatureMatcher* matcher = new OFFeatureMatcher(true, imgs, img_pts, mattes);
    vector<vector<DMatch>> matches;
    matches.resize(imgs.size()-1);
    for (int i = 0; i < imgs.size()-1; i++) {
        matcher->registration(i, i+1, warped_imgs[i], warped_mattes[i]);
    }
    warped_imgs[imgs.size()-1] = imgs[imgs.size()-1].clone();
    warped_mattes[mattes.size()-1] = mattes[mattes.size()-1].clone();

    
    //calculate remats
    remats.resize(imgs.size()-1);
    for (int i = 0; i < imgs.size()-1; i++) {
        remats[i] = warped_imgs[i] - imgs[i+1];
        cvtColor(remats[i], remats[i], CV_BGR2GRAY);
//        imshow("re", remats[i]);
//        waitKey(0);
    }
    
    printf("calculate optical flow cost: %lf\n", (getTickCount()-t0)/getTickFrequency());
}

void App::changeShowState(){
    printf("currentstate:%d\n",currentShowState);
    currentShowState = (currentShowState+1) % 4;
    showImg();
}

void App::start(const vector<string>& trained){
    if (remats.empty()) {
        cerr<<"you need calculate optical flow first!"<<endl;
        exit(1);
    }
    
//    FileStorage p("prob.xml", FileStorage::WRITE);
//    FileStorage c("conf.xml", FileStorage::WRITE);
    classifier = new CombinedClassifier(trained);
    for (int i = 1; i < imgs.size(); i++) {
        Mat UDCprob,UDCconf,raw_dist;
        computeRawDist(warped_mattes[i-1], raw_dist);
        processUDC(imgs[i], warped_mattes[i-1], raw_dist, UDCprob, UDCconf);
        
        Mat localprob,localconf;
        processLC(imgs[i], warped_mattes[i-1], raw_dist, localprob, localconf);
        
        Mat globalprob,globalconf;
        processGC(imgs[i], warped_mattes[i-1], raw_dist,globalprob, globalconf);
        

        Mat shapeprob,shapeconf;
        processSP(imgs[i], warped_mattes[i-1], raw_dist, shapeprob, shapeconf);
        
        Mat errordensity;
        processRegistraionError(remats[i-1], errordensity);
        
        
        //debug
        

//        cout<<globalconf.at<double>(10,371)<<endl;
//        cout<<0.5 + globalconf.at<double>(10,371)*(globalprob.at<double>(10,371)-0.5)<<endl;
//        waitKey(0);
        Mat finalprob(imgs[0].size(),CV_64FC1);
        Mat finalconf(imgs[0].size(),CV_64FC1);
        for (int dx = 0; dx < imgs[0].rows; dx++) {
            for (int dy = 0; dy < imgs[0].cols; dy++) {
                featureVector v;
                v.ru = 0.5 + UDCconf.at<double>(dx,dy)*(UDCprob.at<double>(dx,dy)-0.5);
                v.rl = 0.5 + localconf.at<double>(dx,dy)*(localprob.at<double>(dx,dy)-0.5);
                v.rg = 0.5 + globalconf.at<double>(dx,dy)*(globalprob.at<double>(dx,dy)-0.5);
                v.rs = 0.5 + shapeconf.at<double>(dx,dy)*(shapeprob.at<double>(dx,dy)-0.5);
                v.e = errordensity.at<double>(dx,dy);
//                if (isnan(classifier->prob(v))) {
//                    cout<<classifier->prob(v);
//                }
                finalprob.at<double>(dx,dy) = classifier->prob(v);
                finalconf.at<double>(dx,dy) = classifier->conf(v);
            }
        }
        
        
        //debug
//        imshow("UDCprob", UDCprob);
//        imshow("localprob", localprob);
//        imshow("globalprob", globalprob);
//        imshow("shapeprob", shapeprob);
//        imshow("errorden", errordensity);
//        imshow("ground truth", mattes[i]);
//        imshow("src", imgs[i]);
//        imshow("finalprob", finalprob);
//        imshow("finalconf", finalconf);
        
//        Mat median,mean,gau;
//        finalprob.convertTo(finalprob, CV_32FC1);
//        medianBlur(finalprob, median, 5);
//        blur(finalprob, mean, Size(5,5),Point(-1,-1));
//        GaussianBlur(finalprob, gau, Size(5,5), 0,0);
//        imshow("median", median);
//        imshow("mean", mean);
//        imshow("gaussian", gau);
        
//        refineProb(finalprob);
//        imshow("refinde_prob", finalprob);
        imshow("prob", finalprob);
        Mat dst;
        combinedConfidenceMap(finalprob, finalconf, dst);
        imshow("combined", dst);
        imshow("conf", finalconf);
        dst.convertTo(dst, CV_32FC1);
        Mat ur;
        threshold(dst, ur, 0.5, 1.0, CV_THRESH_BINARY);
//        cout<<ur<<endl;
//        cout<<(int)ur.at<uchar>(2,4);
//        imshow("unknow region", ur*255);
//        waitKey(0);
        Mat trimap(ur.size(),CV_32SC1);
        //construct trimap
        for (int dx = 0; dx < ur.rows; dx++) {
            for (int dy = 0; dy < ur.cols; dy++) {
                if (ur.at<float>(dx,dy)==1) { //known region
                    if (finalprob.at<double>(dx,dy)<0.4) {//background
                        trimap.at<int>(dx,dy)=2;
                    }
                    else{//foreground
                        trimap.at<int>(dx,dy)=1;
                    }
                }
                else{ //unknow region
                    trimap.at<int>(dx,dy)=0;
                }
            }
        }
        cout<<trimap;
        trimap.convertTo(trimap, CV_8UC1);
        imshow("tri", trimap*100);
        waitKey(0);
        Mat solvedMatte;
        solveMatte(imgs[i], trimap, finalprob, finalconf, solvedMatte);
        
        waitKey(0);
        Mat cut;
        getCutout(imgs[i], finalprob, 0.6, cut);
        
//        imshow("result", cut);
//        waitKey(0);
        
        final.push_back(cut);
        output_probs.push_back(finalprob);
        output_confs.push_back(finalconf);
        imwrite("../../result/"+to_string(i)+".jpg", cut);
//        p<<"prob"+to_string(i)<<finalprob;
//        c<<"conf"+to_string(i)<<finalconf;
    }
    
//    for (int i = 0; i < output_probs.size(); i++) {
//        p<<"prob"+to_string(i)<<output_probs[i];
//        c<<"conf"+to_string(i)<<output_confs[i];
//    }
//    p.release();
//    c.release();
}

void App::exportimg(const vector<cv::Mat> &imgs, string path){
    for (int i = 0; i < imgs.size(); i++) {
        imwrite(path+to_string(i)+".jpg", imgs[i]);
    }
}

void App::testUDC(){
    Mat valid(imgs[0].rows, imgs[0].cols,CV_8UC1);
    valid.setTo(255);
    Mat a,b,dist;
    imshow("src", imgs[1]);
    imshow("ground truth", warped_mattes[1]);
    computeRawDist(warped_mattes[0], dist);
    processUDC(imgs[1], warped_mattes[0], dist, a, b);
}


void App::testLocal(){
    vector<int> a;
    Mat train(15,3,CV_32FC1);
    train.setTo(1.5);
    //LocalClassifier k(Vec3d(12,0,0), train, a);
    Mat b,c,dist;
    computeRawDist(warped_mattes[1], dist);
    processLC(imgs[2], warped_mattes[1], dist, b, c);
}

void App::testGlobal(){
    Mat a,b,dist;
    computeRawDist(warped_mattes[0], dist);
    processGC(imgs[1], warped_mattes[0], dist, a, b);
}

void App::testShape(){
    Mat a,b,dist;
    computeRawDist(warped_mattes[0], dist);
    processSP(imgs[1], warped_mattes[0], dist, a, b);
}

void App::testlearn(){
    //string j = "d";
    CombinedClassifier* g = new CombinedClassifier("wtf");
}

void App::maskbypreviousframe(){
    vector<Mat> final;
    for (int i = 1; i < imgs.size(); i++) {
        Mat cut;
        getCutout(imgs[i], warped_mattes[i-1],0.1, cut);
        final.push_back(cut);
    }
    exportimg(final, "../../result/bear_warp/");
}



void App::clear(){
    imgs.clear();
    mattes.clear();
    warped_imgs.clear();
    warped_mattes.clear();
    remats.clear();
}

void App::creategroundtruth(){
    for (int i = 0; i < imgs.size(); i++) {
        Mat gt;
        Mat prob = mattes[i].clone();
        prob.convertTo(prob, CV_64FC1);
        prob = prob/255;
        getCutout(imgs[i], prob, 0.1, gt);
        imwrite("../../result/"+to_string(i)+".jpg", gt);
    }
}






