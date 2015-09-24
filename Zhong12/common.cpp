//
//  common.cpp
//  Zhong12
//
//  Created by DarkTango on 8/29/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "common.h"
#include "Eigen/Sparse"

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
    cutout.create(src.size(), CV_8UC4);
    Mat c[3];
    split(src, c);
    vector<Mat> r;
    r.push_back(c[0]);
    r.push_back(c[1]);
    r.push_back(c[2]);
    r.push_back(prob);
    merge(r, cutout);
 
}

void getCutout2(const Mat& src, const Mat& prob, Mat& cutout){
    cutout = src.clone();
    Mat p;
    if (prob.type()==CV_8U) {
        prob.convertTo(p, CV_64F);
        p/=255;
    }
    else{
        prob.convertTo(p, CV_64F);
    }
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cutout.at<Vec3b>(i,j)=p.at<double>(i,j)*cutout.at<Vec3b>(i,j);
        }
    }
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

double sigmoid(double v){
    return 1.0/(1+exp(-v));
}


bool isboundary(const Mat& constmap, int row, int col){
    assert(constmap.type()==CV_32F);
    if (constmap.at<float>(row,col)==1) {
        
        
        if (row-1>=0) {
            if (constmap.at<float>(row-1,col)==0) {
                return true;
            }
        }
        if (row+1<=constmap.rows-1) {
            if (constmap.at<float>(row+1,col)==0) {
                return true;
            }
        }
        if (col-1>=0) {
            if (constmap.at<float>(row,col-1)==0) {
                return true;
            }
        }
        if (col+1<=constmap.cols) {
            if (constmap.at<float>(row,col+1)==0) {
                return true;
            }
        }
    }
    return false;
}



//trimap = 0 means unknown 1 = foreground 2 = background
const double lambdaS = 20;
void solveMatte(const Mat& src, const Mat& constmap, const Mat& constval, const Mat& prob, const Mat& conf, Mat& dst){
    int64 t0 = getTickCount();

    int matsize = src.rows*src.cols;
    
    SpMat laplacian(matsize, matsize);
    Mat img = src.clone();
    img.convertTo(img, CV_32F);
    
    
    SpMat D(laplacian.rows(), laplacian.cols());
    
    Mat boundary(constmap.size(),CV_32F);
    boundary.setTo(0);
    for (int i = 0; i < boundary.rows; i++) {
        for (int j = 0; j < boundary.cols; j++) {
            if (isboundary(constmap, i, j)) {
                boundary.at<float>(i,j) = 1;
            }
        }
    }
//    imshow("boundary", boundary);
    Mat foreboundary = boundary.mul(constval);

    getL(img, boundary, laplacian);
    
    //by Zhong's method
//    vector<Td> vals;
//    int len = laplacian.rows();
//    
//    Eigen::VectorXd b(len);
//    int count = 0;
//    for (int i = 0; i < constmap.cols; i++) {
//        for (int j = 0; j < constmap.rows; j++) {
//            double lambdaT = conf.at<double>(j,i);
//            double lambdaC = isboundary(constmap, j, i)?100:0;
//            
//            
//            double val = lambdaC+lambdaT;
//            b[count] = lambdaT*sigmoid(prob.at<double>(j,i))+lambdaC*foreboundary.at<float>(j,i);
//            Td t(count,count,val);
//            vals.push_back(t);
//            count++;
//        }
//    }
//    D.setFromTriplets(vals.begin(), vals.end());
//    SpMat A = lambdaS*laplacian + D;
    
    //by my method
//    vector<Td> vals;
//    int len = laplacian.rows();
//    int count = -1;
//    for (int i = 0; i < constmap.cols; i++) {
//        for (int j = 0; j < constmap.rows; j++) {
//            count++;
//            if (boundary.at<float>(j,i)==0) {
//                continue;
//            }
//            Td t(count,count,1);
//            
//            vals.push_back(t);
//        }
//    }
//    D.setFromTriplets(vals.begin(), vals.end());
//    
//    double lambda = 100;
//    Eigen::VectorXd b(len);
//    
//    count = 0;
//    for (int i = 0; i < constmap.cols; i++) {
//        for (int j = 0; j < constmap.rows; j++) {
//            b[count] = (double)boundary.at<float>(j,i)*lambda*foreboundary.at<float>(j,i);
//
//            count++;
//        }
//    }
//    SpMat A = laplacian+lambda*D;
//    
//    Eigen::SimplicialCholesky<SpMat> sol;  // performs a Cholesky factorization of A
//    sol.compute(A);
//    Eigen::VectorXd x = sol.solve(b);
    
    //by origin method
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
    sol.compute(A);
    Eigen::VectorXd x = sol.solve(b);
    
    

    
    dst.create(src.size(), CV_64FC1);
    count = 0;
    for (int i = 0; i < dst.cols; i++) {
        for (int j = 0; j < dst.rows; j++) {
            if(x[count]<0){
                x[count]=0;
            }
            if (x[count]>1) {
                x[count]=1;
            }
            dst.at<double>(j,i) = x[count];
            if (constval.at<float>(j,i)>0) {
                dst.at<double>(j,i) = 1;
            }
            else{
                if (constmap.at<float>(j,i)>0) {
                    dst.at<double>(j, i) = 0;
                }
            }
            count++;
        }
    }
    cout<<"solve matte cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;

    
    
//    imshow("alp", dst);
//
//    waitKey(0);
    
    
}

const int winStep = 1;
constexpr int winLenth = 2*winStep+1;
const double epsilon = 0.0000001;
constexpr int neb_size = winLenth*winLenth;

//trimap = 0 means unknown region
void getL(const Mat& src, const Mat& trimap, SpMat& laplacian){
    
    cvtColor(src, src, CV_BGR2RGB);
    //cout<<INT_MAX<<endl;
    assert((laplacian.rows()==laplacian.cols())&&(laplacian.cols()==(src.cols*src.rows)));
    
    Mat imgidx(src.size(),CV_32SC1);
    
    if (src.type()!=CV_32F) {
        src.convertTo(src, CV_32F);
    }
    
    int count = 0;
    for (int i = 0; i < imgidx.cols; i++) {
        for (int j = 0; j <imgidx.rows; j++) {
            imgidx.at<int>(j,i)=count;
            //printf("%d\n",imgidx.at<int>(i,j));
            count++;
        }
    }
    Mat constm;
    erode(trimap, constm, Mat());

    
    vector<Td> coeffs;
    
    for (int j = winStep; j < src.cols-winStep; j++) {
        for (int i = winStep; i < src.rows-winStep; i++) {
            if (constm.at<float>(i,j)!=0) {
                continue;
            }
            
            Rect wk(j-winStep, i-winStep, winLenth, winLenth);
            Mat winI = src(wk).clone();
            Mat winIdx = imgidx(wk).clone();
            winI = winI.t();
            winI = winI.reshape(1, winLenth*winLenth);
//            cout<<winIdx<<endl;
           // cout<<"winI: "<<winI<<endl;
            
            Mat covMat,meanMat;
            calcCovarMatrix(winI, covMat, meanMat, CV_COVAR_ROWS|CV_COVAR_NORMAL);
//            cout<<"sample: "<<winI<<endl;
//            cout<<"cov: "<<covMat<<endl;
            
//            cout<<"winI*winI: "<<winI.t()*winI<<endl;
//            cout<<"winmu: "<<meanMat.t()*meanMat<<endl;
//            cout<<"3term: "<<epsilon/neb_size*Mat::eye(3, 3, CV_32FC1)<<endl;
            meanMat.convertTo(meanMat, CV_64F);
            winI.convertTo(winI, CV_64F);
            

            Mat cc = winI.t()*winI/neb_size-meanMat.t()*meanMat+epsilon/neb_size*Mat::eye(3, 3, CV_64FC1);
            Mat win_var = cc.inv();

            

            for (int cc = 0; cc < winI.rows; cc++) {
                winI.row(cc) = winI.row(cc)-meanMat;
            }
            
            Mat vals = (1+winI*win_var*winI.t())/(float)neb_size;
            

            winIdx = winIdx.t();
            winIdx = winIdx.reshape(1,1);
            Mat colid,rowid;
            repeat(winIdx, 1, neb_size, rowid);
            repeat(winIdx, neb_size, 1, colid);
            colid = colid.t();
            colid = colid.reshape(1, 1);
            
            vals = vals.t();
            vals = vals.reshape(1, 1);
            for (int cc = 0; cc < neb_size*neb_size; cc++) {
                Td _tmp(rowid.at<int>(0,cc),colid.at<int>(0,cc),vals.at<double>(0,cc));
                
                coeffs.push_back(_tmp);
            }
            
        }
    }
    


    
    laplacian.setFromTriplets(coeffs.begin(), coeffs.end());
    
    vector<Td> sumL,sumT;
    int row = laplacian.rows();
    SpMat tmp = laplacian.transpose();
    for (int i = 0; i < row; i++) {
        double sum = tmp.col(i).sum();
        
        Td tmp(i,i,sum);
        sumL.push_back(tmp);
        //printf("%d %lf\n",i,sum);
    }
    
    
    SpMat diag(laplacian.rows(),laplacian.cols());
    diag.setFromTriplets(sumL.begin(), sumL.end());
    laplacian = diag - laplacian;

    return;

}

void loadSPmat(string filename, SpMat& sp){
    FILE* stream;
    stream = fopen(filename.c_str(), "r");
    if (stream==NULL) {
        cerr<<"no file: "<<filename<<endl;
        exit(0);
    }
    vector<Td> data;
    while (!feof(stream)) {
        int a,b;
        double val;
        fscanf(stream, "         (%d,%d)             %lf\n",&a,&b,&val);
        Td tmp(a-1,b-1,val);
        data.push_back(tmp);
        //printf("%d %d %lf\n",a,b,val);
    }
    fclose(stream);
    sp.setFromTriplets(data.begin(), data.end());
}


        
void compareMat(SpMat& mat1, SpMat& mat2){
    SpMat minus = mat1 - mat2;
    for (int k=0; k<minus.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(minus,k); it; ++it)
        {
            if (fabs(it.value())>1) {
                cout<<"row: "<<it.row()<<" col: "<<it.col()<<" val: "<<it.value()<<endl;
            }
        }
}



