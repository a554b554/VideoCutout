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
        Mat m = imread(fname,0);
        if (m.empty()) {
            continue;
        }
        mattes.push_back(m);
        //        imshow("matte", mattes[mattes.size()-1]);
        //        waitKey(0);
    }
    closedir(dp);
    
}

void loadmatte(string dirname, vector<Mat>& mattes){
    string path = dirname + "/";
    
    
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
        Mat img = imread(fname,0);
        if (img.empty()) {
            continue;
        }
        mattes.push_back(img);
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

void App::computeOpitcalFlow(const Mat& srcimg, const Mat& dstimg, const Mat& srcmatte, Mat& warped_matte, Mat& warped_img){
    int64 t0 = getTickCount();
    vector<Mat> _imgs;
    vector<Mat> _mattes;
    _imgs.push_back(srcimg);
    _imgs.push_back(dstimg);
    _mattes.push_back(srcmatte);
    _mattes.push_back(srcmatte);
    vector<vector<KeyPoint>> img_pts;
    OFFeatureMatcher matcher(true, _imgs, img_pts, _mattes);
    
    matcher.registration(0, 1, warped_img, warped_matte);
    //debug
    cout<<"compute optical flow cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;

//    imshow("matte", warped_matte+warped_img);
//    imshow("align + after img", 0.5*imgs[0]+0.5*warped_img);
//    imshow("align + before img", 0.5*imgs[0]+0.5*imgs[1]);
//    waitKey(0);
    
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
//        imshow("prob", finalprob);
        Mat dst;
        combinedConfidenceMap(finalprob, finalconf, dst);
//        imshow("combined", dst);
//        imshow("conf", finalconf);
        dst.convertTo(dst, CV_32FC1);
        Mat ur;
        threshold(dst, ur, 0.2 , 1.0, CV_THRESH_BINARY);
//        cout<<ur<<endl;
//        cout<<(int)ur.at<uchar>(2,4);
//        imshow("unknow region", ur*255);
//        waitKey(0);
        //Mat trimap(ur.size(),CV_32SC1);
        Mat constmap(ur.size(),CV_32FC1);
        Mat constval(ur.size(),CV_32FC1);
        //construct trimap
        constmap.setTo(0);
        constval.setTo(0);
        
        
        for (int dx = 0; dx < ur.rows; dx++) {
            for (int dy = 0; dy < ur.cols; dy++) {
                if (ur.at<float>(dx,dy)==1) { //known region
                    if (finalprob.at<double>(dx,dy)<0.4) {//background
                        constmap.at<float>(dx,dy)=1;
                    }
                    else{//foreground
                        constmap.at<float>(dx,dy)=1;
                        constval.at<float>(dx,dy)=1;
                    }
                }

            }
        }
        
       
//        imshow("constmap", constmap);
//        imshow("constval", constval);
        Mat constval_cut,known;
        getCutout2(imgs[i], constval, constval_cut);
        getCutout2(imgs[i], constmap, known);
//        imshow("constval_cut", constval_cut);
//        imshow("known", known);
        //waitKey(0);
        Mat solvedMatte;
        solveMatte(imgs[i], constmap, constval, finalprob, finalconf, solvedMatte);
        
//        imshow("matte", solvedMatte);
        
        Mat cut,probcut;
        getCutout2(imgs[i], solvedMatte, cut);
        getCutout2(imgs[i], finalprob, probcut);
        
//        imshow("result", cut);
//        imshow("probcut", probcut);
//        waitKey(0);
        final.push_back(cut);
        output_probs.push_back(finalprob);
        output_confs.push_back(finalconf);
        imwrite("../../result/"+to_string(i)+".png", cut);
        solvedMatte=solvedMatte*255;
        solvedMatte.convertTo(solvedMatte, CV_8U);
        imwrite("../../result/"+to_string(i)+"_alpha.png", solvedMatte);
        imwrite("../../result/"+to_string(i)+"_alpha_p.png", probcut);
        
        
//        warped_mattes[i] = solvedMatte.clone();
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

void App::start2(const vector<string>& trained){

    Mat currentmatte = mattes[0];
    classifier = new CombinedClassifier(trained);
    for (int i = 1; i < imgs.size(); i++) {
        //compute optical flow
        Mat warped_matte,warped_img;
        computeOpitcalFlow(imgs[i-1], imgs[i], currentmatte, warped_matte, warped_img);
        
        
        
        Mat UDCprob,UDCconf,raw_dist;
        computeRawDist(warped_matte, raw_dist);
        processUDC(imgs[i], warped_matte, raw_dist, UDCprob, UDCconf);
        
        Mat localprob,localconf;
        processLC(imgs[i], warped_matte, raw_dist, localprob, localconf);
        
        Mat globalprob,globalconf;
        processGC(imgs[i], warped_matte, raw_dist,globalprob, globalconf);
        
        
        Mat shapeprob,shapeconf;
        processSP(imgs[i], warped_matte, raw_dist, shapeprob, shapeconf);
        
        Mat errordensity,re;
        re = warped_img - imgs[i];
        cvtColor(re, re, CV_BGR2GRAY);
        processRegistraionError(re, errordensity);
        
        
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
//                imshow("errorden", errordensity);
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
        //        imshow("prob", finalprob);
        Mat dst;
        combinedConfidenceMap(finalprob, finalconf, dst);
                imshow("combined", dst);
        //        imshow("conf", finalconf);
        dst.convertTo(dst, CV_32FC1);
        Mat ur;
        threshold(dst, ur, 0.6 , 1.0, CV_THRESH_BINARY);
        //        cout<<ur<<endl;
        //        cout<<(int)ur.at<uchar>(2,4);
//        imshow("unknow region", ur*255);
        //        waitKey(0);
        //Mat trimap(ur.size(),CV_32SC1);
        Mat constmap(ur.size(),CV_32FC1);
        Mat constval(ur.size(),CV_32FC1);
        //construct trimap
        constmap.setTo(0);
        constval.setTo(0);
        
        
        for (int dx = 0; dx < ur.rows; dx++) {
            for (int dy = 0; dy < ur.cols; dy++) {
                if (ur.at<float>(dx,dy)==1) { //known region
                    if (finalprob.at<double>(dx,dy)<0.4) {//background
                        constmap.at<float>(dx,dy)=1;
                    }
                    else{//foreground
                        constmap.at<float>(dx,dy)=1;
                        constval.at<float>(dx,dy)=1;
                    }
                }
                
            }
        }
        
        
//        imshow("constmap", constmap);
//        imshow("constval", constval);
        Mat constval_cut,known;
        getCutout2(imgs[i], constval, constval_cut);
        getCutout2(imgs[i], constmap, known);
        imshow("constval_cut", constval_cut);
        imshow("known", known);
        //waitKey(0);
        Mat solvedMatte;
        solveMatte(imgs[i], constmap, constval, finalprob, finalconf, solvedMatte);
        
        imshow("matte", solvedMatte);
        waitKey(0);
        Mat cut,probcut;
        getCutout2(imgs[i], solvedMatte, cut);
        getCutout2(imgs[i], finalprob, probcut);
        
        imshow("result", cut);
        imshow("probcut", probcut);
        //waitKey(0);
        final.push_back(cut);
        output_probs.push_back(finalprob);
        output_confs.push_back(finalconf);
        imwrite("../../result/"+to_string(i)+".png", cut);
        solvedMatte=solvedMatte*255;
        solvedMatte.convertTo(solvedMatte, CV_8U);
        imwrite("../../result/mr"+to_string(i)+".png", solvedMatte);
//        imwrite("../../result/"+to_string(i)+"_alpha_p.png", probcut);
        
        imshow("current", currentmatte);
        imshow("propagate", solvedMatte-currentmatte);
//        waitKey(0);
        currentmatte = solvedMatte.clone();
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

static vector<Point> fmattee,bmattee;
static bool setf = false;
static bool setb = false;
void onMouse(int event,int x,int y,int,void*){
    if(setf)
    {
        cout<<"setting F: "<<Point(x,y)<<endl;
        fmattee.push_back(Point(x,y));
    }
    if (setb) {
        cout<<"setting B: "<<Point(x,y)<<endl;
        bmattee.push_back(Point(x,y));
    }
    if(event==CV_EVENT_LBUTTONDOWN){
        cout<<"begin F: "<<Point(x,y)<<endl;
        setf = true;
        fmattee.push_back(Point(x,y));
     
    }
    if(event==CV_EVENT_LBUTTONUP)
    {
        cout<<"end F: "<<Point(x,y)<<endl;
        setf = false;
    }
    if (event==CV_EVENT_RBUTTONDOWN) {
        cout<<"begin B: "<<Point(x,y)<<endl;
        setb = true;
        bmattee.push_back(Point(x,y));
    }
    if (event==CV_EVENT_RBUTTONUP) {
        cout<<"end B: "<<Point(x,y)<<endl;
        setb = false;
    }
}

void App::start3(const vector<string>& trained){
    
    classifier = new CombinedClassifier(trained);
    vector<Mat> right,left;
    loadmatte("../../result/BB_forward", right);
    loadmatte("../../result/BB_backward", left);


    namedWindow("main");
    cvSetMouseCallback("main", onMouse);
    
    for (int i = right.size()-1; i>=0; i--) {
        Mat current = right[i];
        Mat out,show;
        getCutout2(imgs[i], current, out);
        show = imgs[i].clone();
        while (1) {
            
            if (waitKey(10)=='a') {
                for (int i = 0; i < fmattee.size(); i++) {
                    circle(current, fmattee[i], 5, 255, -1);
                }
                for (int i = 0; i < bmattee.size(); i++) {
                    circle(current, bmattee[i], 5, 0, -1);
                }
                fmattee.clear();
                bmattee.clear();
            }
            if (waitKey(10)=='q') {
                //imwrite("../../"+to_string(i)+".png", out);
                cout<<"break!"<<endl;
                break;
            }
            if (waitKey(10)=='b') {
                show = imgs[i].clone();
                grabCut(show, current, Rect(), Mat(), Mat(), 5, GC_INIT_WITH_MASK);
            }
            
            getCutout2(imgs[i], current, out);
        
            imshow("main", out);
            imshow("curr", current);
            //imshow("grabcut", show);
        }
        imshow("main", out);
        imshow("curr", current);
        //imshow("dif", current-right[i-1]);
        Mat ds = current - right[i-1];
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        findContours( ds, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        vector<int> good;
        double minArea = 500;
        Mat dscp;
        cvtColor(ds, dscp, CV_GRAY2BGR);
        vector<Rect> boundingboxes;
        for (int k = 0; k < contours.size(); k++) {
            if (contourArea(contours[k]) < minArea) {
                good.push_back(0);
                continue;
            }
            good.push_back(1);
            drawContours(dscp, contours, k, Scalar(255,255,0));
            Rect r = boundingRect(contours[k]);
            if (r.width>=21&&r.height>=21) {
                boundingboxes.push_back(r);
                rectangle(out, r, Scalar(0,255,255));
            }

            
        }
        
        //debug
        imshow("bounding box", out);
        imshow("contour", dscp);
        imshow("diff", ds);
        Mat previous;
        getCutout2(imgs[i-1], right[i-1], previous);
        imshow("previous", previous);
        waitKey(0);
        
        
        Mat warped_matte,warped_img;
        computeOpitcalFlow(imgs[i], imgs[i-1], current, warped_matte, warped_img);
        for (int k = 0; k < boundingboxes.size(); k++) {
            Mat boxmatte,boximg;
            boximg = imgs[i](boundingboxes[k]).clone();
            boxmatte = warped_matte(boundingboxes[k]).clone();
            
            
            
            
            
            Mat UDCprob,UDCconf,raw_dist;
            computeRawDist(boxmatte, raw_dist);
            processUDC(boximg, boxmatte, raw_dist, UDCprob, UDCconf);
            
            Mat localprob,localconf;
            processLC(boximg, boxmatte, raw_dist, localprob, localconf);
            
            Mat globalprob,globalconf;
            processGC(boximg, boxmatte, raw_dist,globalprob, globalconf);
            
            
            Mat shapeprob,shapeconf;
            processSP(boximg, boxmatte, raw_dist, shapeprob, shapeconf);
            
            Mat errordensity,re;
            re = warped_img(boundingboxes[k]) - boximg;
            cvtColor(re, re, CV_BGR2GRAY);
            processRegistraionError(re, errordensity);
            
            
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
                    
                    
                    finalprob.at<double>(dx,dy) = classifier->prob(v);
                    finalconf.at<double>(dx,dy) = classifier->conf(v);
                }
            }

            //debug
            imshow("UDCprob", UDCprob);
            imshow("localprob", localprob);
            imshow("globalprob", globalprob);
            imshow("shapeprob", shapeprob);
            imshow("errorden", errordensity);
            imshow("boxmatte", boxmatte);
            imshow("boximg", boximg);
            imshow("finalprob", finalprob);
            imshow("finalconf", finalconf);
            waitKey(0);
            
            
        }
        
    }
    
    
    //refinement(right, left);
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


void App::reverse(){
    std::reverse(imgs.begin(), imgs.end());
    std::reverse(mattes.begin(), mattes.end());
}


void App::refinement(vector<cv::Mat> &right, vector<cv::Mat> &left){
    vector<Mat> finalmat;
    for (int i = 0; i < right.size(); i++) {
        finalmat.push_back(right[i]);
    }
}

