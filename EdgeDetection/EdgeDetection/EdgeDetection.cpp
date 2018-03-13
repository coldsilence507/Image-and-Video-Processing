//
//  EdgeDetection.cpp
//  EdgeDetection
//
//  Created by TJLin on 2/13/18.
//  Copyright © 2018 TJLin. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <stack>
#include <iomanip>

using namespace cv;
using namespace std;

//gaussian smoothing
//Gaussian Computation
//input: number of sigma, sigma, mu
//output: Gaussian distribution
double Gaussian(int n, double sigma, int mu = 0)
{
    return 1/sqrt(2*M_PI*sigma)*exp(-pow(n,2)/(2*pow(sigma, 2)));
}

//Gaussian Kernel Computation
//input: kernel size, sigma, mu
//output: Gaussian Kernel in X direction
Mat genGaussKernelX(int n, double sigma, int mu = 0){
    try{
        if(n < 5 || n%2 != 1) throw 1;
    }
    catch(int n){
        cerr<<"Kernel size should not be less than 5 or even.";
    }
    
    Mat GaussX(1, n, CV_32F);
    int mid = n/2;
    
    for(int i = 0; i<n; i++){
        GaussX.at<float>(0,i) = Gaussian(i-mid, sigma, mu);
    }
    double sum = cv::sum(GaussX)[0];
    GaussX /= sum;
    return GaussX;
}

//Gaussian Filter
//intput: source image, kernel size, sigma, mu
//
Mat Gaussian_Filter(Mat src, int n, double sigma, double mu = 0){
    Mat kernel;
    int kernel_size;
    kernel_size = n;
    Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth = CV_32F;
    Mat GaussX, GaussY;
    
    GaussX = genGaussKernelX(n, sigma);
    transpose(GaussX, GaussY);
    Mat res = Mat::zeros( src.rows, src.cols, CV_32F );

    filter2D(src, src, ddepth , GaussX, anchor, delta, BORDER_DEFAULT );
    filter2D(src, res, ddepth , GaussY, anchor, delta, BORDER_DEFAULT );
    
    return res;
}

//Canny Enhancement

//X Gradient computation
//input: Source Image after Gaussian smoothing
//output: X gradient matrix
Mat GradComponentX(Mat src){
    Mat filter = (Mat_<int>(1,3)<<-1,0,1);
    Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth = CV_32F;
    Mat res = Mat::zeros( src.rows, src.cols, CV_32F );

    filter2D(src, res, ddepth , filter, anchor, delta, BORDER_DEFAULT );
    return res;
}

//Y Gradient computation
//input: Source Image after Gaussian smoothing
//output: Y gradient matrix
Mat GradComponentY(Mat src){
    Mat filter = (Mat_<int>(3,1)<<-1,0,1);
    Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth = CV_32F;
    Mat res = Mat::zeros( src.rows, src.cols, CV_32F );
    
    filter2D(src, res, ddepth , filter, anchor, delta, BORDER_DEFAULT );
    return res;
}

//Edge Strength Computation
//input: X and Y gradient Matrix
//output: Edge Strength Matrix
Mat EdgeStr(Mat X, Mat Y){
    Mat res = Mat::zeros( X.rows, X.cols, CV_32F );
    
    for(int i=0; i<X.rows; i++){
        for(int j=0; j<X.cols; j++){
            float Jx = X.at<float>(i,j);
            float Jy = Y.at<float>(i,j);
            res.at<float>(i,j) = sqrt(pow(Jx, 2) + pow(Jy, 2));
        }
    }
    return res;
}

//Edge Normal Orientation Computation
//input: X and Y gradient Matrix
//output: Edge Normal Orientation Matrix
Mat EdgeOri(Mat X, Mat Y){
    Mat res = Mat::zeros( X.rows, X.cols, CV_32F );
    
    for(int i=0; i<X.rows; i++){
        for(int j=0; j<X.cols; j++){
            float Jx = X.at<float>(i,j);
            float Jy = Y.at<float>(i,j);
            float ori = atan2(Jy, Jx)*180/M_PI;
            if (ori<0) ori+=180;
            res.at<float>(i,j) = ori;
        }
    }
    return res;
}

//Edge Normal Direction Estimation, to 0,45,90,135
//input: Edge Normal Orientation Matrix
//output: Estimated Edge Normal Orientation Matrix, in 0, 45, 90, 135
Mat DirectionEst(Mat EOri){
    Mat res = Mat::zeros( EOri.rows, EOri.cols, CV_8U );
    for(int i=0; i<EOri.rows;i++){
        for(int j=0; j<EOri.cols; j++){
            if((EOri.at<float>(i,j)>=0 && EOri.at<float>(i,j)<22.5) || (EOri.at<float>(i,j)>157.5 && EOri.at<float>(i,j)<=180))
                res.at<uchar>(i,j)=0;
            else if(EOri.at<float>(i,j)>= 22.5 && EOri.at<float>(i,j) < 67.5)
                res.at<uchar>(i,j)=45;
            else if(EOri.at<float>(i,j)>= 67.5 && EOri.at<float>(i,j) < 112.5)
                res.at<uchar>(i,j)=90;
            else
                res.at<uchar>(i,j)=135;
        }
    }
    
    return res;
}

//Non-Max Suppression
//input: Estimated Edge Normal Orientation Matrix, Edge Strength Matrix
//output: Suppressed matrix I_N
Mat NMSupp(Mat DirEst, Mat EStr){
    Mat res = Mat::zeros( EStr.rows, EStr.cols, CV_32F );
    for(int i=1; i<EStr.rows-1; i++){
        for (int j=1; j<EStr.cols-1;j++){
            int cur = DirEst.at<uchar>(i,j);
            switch (cur){
                case 0:
                {
                    if(EStr.at<float>(i,j) >= EStr.at<float>(i,j-1) && EStr.at<float>(i,j) >= EStr.at<float>(i,j+1))
                        res.at<float>(i, j) = EStr.at<float>(i,j);
                    break;
                }
                case 45:
                {
                    if(EStr.at<float>(i,j) >= EStr.at<float>(i-1,j-1) && EStr.at<float>(i,j) >= EStr.at<float>(i+1,j+1))
                        res.at<float>(i,j) = EStr.at<float>(i,j);
                    break;
                }
                case 90:
                {
                    if(EStr.at<float>(i,j) >= EStr.at<float>(i-1,j) && EStr.at<float>(i,j) >= EStr.at<float>(i+1,j))
                        res.at<float>(i,j) = EStr.at<float>(i,j);
                    break;
                }
                case 135:
                {
                    if(EStr.at<float>(i,j) >= EStr.at<float>(i-1,j+1) && EStr.at<float>(i,j) >= EStr.at<float>(i+1,j-1))
                        res.at<float>(i,j) = EStr.at<float>(i,j);
                    break;
                }
            }
        }
    }
    return res;
}

//pixel validity
bool isvalidpix(int i, int j, int n, int m){
    return i>=0 && j>=0 && i<n && j<m;
}

//Hysteresis Thresholding
//input: Suppressed matrix I_N, Estimated Edge Normal Orientation Matrix，lower threshold, higher threshold
//output: Final result of Canny Edge Detection
Mat Hysteresis_Thres(Mat NMS, Mat DirEst, float Tl, float Th){
    Mat res = Mat::zeros( NMS.rows, NMS.cols, CV_32F );
    Mat visited = Mat::zeros( NMS.rows, NMS.cols, CV_8U );

    //put all strong edge pixels into a stack
    stack<pair<int, int>> Edges;
    for(int i=0; i<NMS.rows;i++){
        for(int j=0; j<NMS.cols;j++){
            if (NMS.at<float>(i,j)>Th){
                pair<int,int> newpair = make_pair(i,j);
                Edges.push(newpair);
            }
        }
    }
    
    //track all weak edges pixels are connected to strong edge pixels
    while(!Edges.empty()){
        pair<int,int> top = Edges.top();
        int r = top.first;
        int c = top.second;
        res.at<float>(r,c) = NMS.at<float>(r,c);
        Edges.pop();
        if(visited.at<uchar>(r,c) == 0){
            int curDir = DirEst.at<uchar>(r,c);
            switch (curDir){
                case 0:{
                    for(int i = -1; i<=1;i+=2){
                        if(isvalidpix(r,c+i,NMS.rows,NMS.cols) && visited.at<uchar>(r,c+i)==0 && NMS.at<float>(r,c+i)>Tl){
                            pair<int,int> newpair = make_pair(r,c+i);
                            Edges.push(newpair);
                        }
                    }
                    visited.at<uchar>(r,c) = 1;
                break;
                }
                case 45:{
                    for(int i = -1; i<=1;i+=2){
                        if(isvalidpix(r-i,c+i,NMS.rows,NMS.cols) && visited.at<uchar>(r-i,c+i)==0 && NMS.at<float>(r-i,c+i)>Tl){
                            pair<int,int> newpair = make_pair(r-i,c+i);
                            Edges.push(newpair);
                        }
                    }
                    visited.at<uchar>(r,c) = 1;
                    break;
                }
                case 90:{
                    for(int i = -1; i<=1;i+=2){
                        if(isvalidpix(r+i,c,NMS.rows,NMS.cols) && visited.at<uchar>(r+i,c)==0 && NMS.at<float>(r+i,c)>Tl){
                            pair<int,int> newpair = make_pair(r+i,c);
                            Edges.push(newpair);
                        }
                    }
                    visited.at<uchar>(r,c) = 1;
                    break;
                }
                case 135:{
                    for(int i = -1; i<=1;i+=2){
                        if(isvalidpix(r+i,c+i,NMS.rows,NMS.cols) && visited.at<uchar>(r+i,c+i)==0 && NMS.at<float>(r+i,c+i)>Tl){
                            pair<int,int> newpair = make_pair(r+i,c+i);
                            Edges.push(newpair);
                        }
                    }
                    visited.at<uchar>(r,c) = 1;
                    break;
                }
            }
        }
    }
    
    return res;
}

//One Function Canny Edge Detection
//input: source image, sigma for Gaussian filter, lower threshold, higher threshold
//output: final result
Mat CannyEdgeDetection(Mat src, double sigma, int Tl, int Th){
    Mat res = Mat::zeros( src.rows, src.cols, CV_32F );
    Mat XGrad = Mat::zeros( src.rows, src.cols, CV_32F );
    Mat YGrad = Mat::zeros( src.rows, src.cols, CV_32F );
    Mat EStr = Mat::zeros( src.rows, src.cols, CV_32F );
    Mat EOri = Mat::zeros( src.rows, src.cols, CV_32F );
    Mat DirEst = Mat::zeros( src.rows, src.cols, CV_8U );
    Mat I_N = Mat::zeros( src.rows, src.cols, CV_32F );
    
    //Canny Enhance
    src = Gaussian_Filter(src, 5, sigma);
    XGrad = GradComponentX(src);
    YGrad = GradComponentY(src);
    EStr = EdgeStr(XGrad, YGrad);
    EOri = EdgeOri(XGrad, YGrad);
    
    //Non-Max Suppression
    DirEst = DirectionEst(EOri);
    I_N = NMSupp(DirEst, EStr);
    
    //Hysteresis Thresholding
    res = Hysteresis_Thres(I_N, DirEst, Tl, Th);
    
    return res;
}

int main() {
    Mat srcimg1, srcimg2, srcimg3, srcimg4, resimg1, resimg2, resimg3, resimg4;
    srcimg1 = imread( "../../src/image1.jpg",CV_LOAD_IMAGE_GRAYSCALE );
    srcimg2 = imread( "../../src/Syracuse_01.jpg",CV_LOAD_IMAGE_GRAYSCALE );
    srcimg3 = imread( "../../src/Syracuse_02.jpg",CV_LOAD_IMAGE_GRAYSCALE );
    srcimg4 = imread( "../../src/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    resimg1 = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    resimg2 = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    resimg3 = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    resimg4 = Mat::zeros( srcimg4.rows, srcimg3.cols, CV_32F );
    
    if( !srcimg1.data || !srcimg2.data || !srcimg3.data)
    {   cout<< "Failed to load source image." << endl;
        return -1; }
    
    double sigma;
    int Tl;
    int Th;
    std::ostringstream out;
    out << setprecision(1) << fixed;

    
    vector<int> para = {CV_IMWRITE_JPEG_QUALITY, 100};
    
    //convert to float for more precise computation
    Mat srcimg1_f = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    Mat srcimg2_f = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    Mat srcimg3_f = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    Mat srcimg4_f = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_32F );
    srcimg1.convertTo(srcimg1_f, CV_32F);
    srcimg2.convertTo(srcimg2_f, CV_32F);
    srcimg3.convertTo(srcimg3_f, CV_32F);
    srcimg4.convertTo(srcimg4_f, CV_32F);
    

    
    //-----------------------------<image 1 Processing>--------------------------------
    
    Mat XGrad = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    Mat YGrad = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    Mat EStr = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    Mat EOri = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    Mat DirEst = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_8U );
    Mat I_N = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_32F );
    

    
    sigma = 1;
    Th = 80;
    Tl = 0.4 * Th;

    //Canny Enhance
    srcimg1_f = Gaussian_Filter(srcimg1_f, 5, sigma);
    XGrad = GradComponentX(srcimg1_f);
    YGrad = GradComponentY(srcimg1_f);
    EStr = EdgeStr(XGrad, YGrad);
    EOri = EdgeOri(XGrad, YGrad);

    //Non-Max Suppression
    DirEst = DirectionEst(EOri);
    I_N = NMSupp(DirEst, EStr);
    
    //Hysteresis Thresholding
    //resimg1 = Hysteresis_Thres(I_N, DirEst, 20, 70);
    
    //One Function canny edge detection
    resimg1 = CannyEdgeDetection(srcimg1, sigma, Tl, Th);
    
    out << sigma;
    //output
//    imwrite("../../src/image1_Gaus_" + out.str() + ".jpg",srcimg1_f, para);
//    imwrite("../../src/image1_Jx_" + out.str() + ".jpg",XGrad, para);
//    imwrite("../../src/image1_Jy_" + out.str() + ".jpg",YGrad, para);
    imwrite("../../src/image1_es_" + out.str() + ".jpg",EStr, para);
//    imwrite("../../src/image1_eo_" + out.str() + ".jpg",EOri, para);
//    imwrite("../../src/image1_eeo_" + out.str() + ".jpg",DirEst, para);
    //imwrite("../../src/image1_nms_" + out.str() + ".jpg",I_N, para);
    imwrite("../../src/image1_" + out.str() + "_" + to_string(Tl) +"_" + to_string(Th) +".jpg", resimg1, para);
    out.clear();
    out.str("");
    //-----------------------------<image 2 Processing>--------------------------------
    
    XGrad = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    YGrad = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    EStr = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    EOri = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    DirEst = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_8U );
    I_N = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_32F );
    

    sigma = 2;
    Th = 30;
    Tl = 0.4 * Th;

    
    //Canny Enhance
    srcimg2_f = Gaussian_Filter(srcimg2_f, 5, sigma);
    XGrad = GradComponentX(srcimg2_f);
    YGrad = GradComponentY(srcimg2_f);
    EStr = EdgeStr(XGrad, YGrad);
    EOri = EdgeOri(XGrad, YGrad);

    //Non-Max Suppression
    DirEst = DirectionEst(EOri);
    I_N = NMSupp(DirEst, EStr);
    
    //Hysteresis Thresholding
    //resimg2 = Hysteresis_Thres(I_N, DirEst, 20, 60);

    //One Function canny edge detection
    resimg2 = CannyEdgeDetection(srcimg2, sigma, Tl, Th);
    
    out << sigma;
    //imwrite("../../src/image2_es.jpg",EStr, para);
    //imwrite("../../src/image2_nms_" + out.str() + ".jpg",I_N, para);
    imwrite("../../src/image2_" + out.str() + "_" + to_string(Tl) +"_" + to_string(Th) +".jpg", resimg2, para);
    out.clear();
    out.str("");

    //-----------------------------<image 3 Processing>--------------------------------
    XGrad = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    YGrad = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    EStr = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    EOri = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    DirEst = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_8U );
    I_N = Mat::zeros( srcimg3.rows, srcimg3.cols, CV_32F );
    
    sigma = 1;
    Th = 70;
    Tl = 0.4*Th;

    
    //Canny Enhance
    srcimg3_f = Gaussian_Filter(srcimg3_f, 5, sigma);
    XGrad = GradComponentX(srcimg3_f);
    YGrad = GradComponentY(srcimg3_f);
    EStr = EdgeStr(XGrad, YGrad);
    EOri = EdgeOri(XGrad, YGrad);

    //Non-Max Suppression
    DirEst = DirectionEst(EOri);
    I_N = NMSupp(DirEst, EStr);
    
    //Hysteresis Thresholding
    //resimg2 = Hysteresis_Thres(I_N, DirEst, 20, 50);
    
    //One Function canny edge detection
    resimg3 = CannyEdgeDetection(srcimg3, sigma, Tl, Th);
    
    out << sigma;
    //imwrite("../../src/image3_es.jpg",EStr, para);
    //imwrite("../../src/image3_nms_" + out.str() + ".jpg",I_N, para);
    imwrite("../../src/image3_" + out.str() + "_" + to_string(Tl) +"_" + to_string(Th) + ".jpg", resimg3, para);
    out.clear();
    out.str("");

    //-----------------------------<image 4 Processing>--------------------------------
    XGrad = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_32F );
    YGrad = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_32F );
    EStr = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_32F );
    EOri = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_32F );
    DirEst = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_8U );
    I_N = Mat::zeros( srcimg4.rows, srcimg4.cols, CV_32F );
    
    sigma = 1;
    Th = 80;
    Tl = 20;
    //Canny Enhance
    srcimg4_f = Gaussian_Filter(srcimg4_f, 5, sigma);
    XGrad = GradComponentX(srcimg4_f);
    YGrad = GradComponentY(srcimg4_f);
    EStr = EdgeStr(XGrad, YGrad);
    EOri = EdgeOri(XGrad, YGrad);
    
    //Non-Max Suppression
    DirEst = DirectionEst(EOri);
    I_N = NMSupp(DirEst, EStr);
    
    //Hysteresis Thresholding
    //resimg1 = Hysteresis_Thres(I_N, DirEst, 20, 70);
    
    //One step canny edge detection
    resimg4 = CannyEdgeDetection(srcimg4, sigma, Tl, Th);
    
    out << sigma;
    //output
    //imwrite("../../src/image1_es.jpg",EStr, para);
    //imwrite("../../src/image4_nms_" + out.str() + ".jpg",I_N, para);
    imwrite("../../src/image4_" + out.str() + "_" + to_string(Tl) +"_" + to_string(Th) + ".jpg", resimg4, para);
    out.clear();
    out.str("");

    return 0;
}
