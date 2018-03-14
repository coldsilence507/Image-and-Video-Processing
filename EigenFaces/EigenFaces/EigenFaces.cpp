//
//  main.cpp
//  EigenFaces
//
//  Created by TJLin on 3/13/18.
//  Copyright © 2018 TJLin. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <stack>
#include <iomanip>
#include <unordered_set>
#include <chrono>


using namespace std;

class Training{
public:

    Training(){}
    ~Training(){}
    
    vector<cv::Mat>& trainingimages(){return trainingimages_;}
    vector<cv::Mat> trainingimages()const{return trainingimages_;}
    Training& trainingimages(vector<cv::Mat> imgs){trainingimages_=imgs;return *this;}
    
    int& row(){return row_;}
    int row()const{return row_;}
    Training& row(int r){row_ = r; return *this;}
    
    int& col(){return col_;}
    int col()const{return col_;}
    Training& col(int c){col_ = c; return *this;}
    
    int& size(){return size_;}
    int size()const{return size_;}
    Training& size(int n){size_ = n; return *this;}
    
    int& imgsize(){return imgsize_;}
    int imgsize()const{return imgsize_;}
    Training& imgsize(int n){imgsize_ = n; return *this;}
    
    cv::Mat& averageface(){return averageface_;}
    cv::Mat averageface()const{return averageface_;}
    
    cv::Mat& eigenvector1(){return eigenvector1_;}
    cv::Mat eigenvector1()const{return eigenvector1_;}
    
    cv::Mat& U(){return U_;}
    cv::Mat U()const{return U_;}
    
    cv::Mat& eigenvector3(){return eigenvector3_;}
    cv::Mat eigenvector3()const{return eigenvector3_;}

    cv::Mat& eigenfaces(){return eigenfaces_;}
    cv::Mat eigenfaces()const{return eigenfaces_;}
    
    vector<cv::Mat>& topeigenfaces(){return topeigenfaces_;}
    vector<cv::Mat> topeigenfaces()const{return topeigenfaces_;}

    void initialize(vector<cv::Mat> trainingset);
    void computeAverageFace();
    void computeMatX();
    
    void PCA1();
    void PCA2();
    void PCA3();
    
    void ComputeEigenfaces();
    void FindSigEigenfaces(int n);

private:
    vector<cv::Mat> trainingimages_;
    cv::Mat averageface_;
    int row_;
    int col_;
    int size_;
    int imgsize_;
    cv::Mat eigenvalue1_;
    cv::Mat eigenvector1_;
    cv::Mat MatX_;
    cv::Mat U_;
    cv::Mat SIGMA_;
    cv::Mat VT_;
    cv::Mat eigenvalue3_;
    cv::Mat eigenvector3_;
    
    cv::Mat eigenfaces_;
    vector<cv::Mat> topeigenfaces_;

};

void Training::initialize(vector<cv::Mat> trainingset)
{
    if(trainingset.empty())
        return;
    trainingimages_ = trainingset;
    row_ = trainingset[0].rows;
    col_ = trainingset[0].cols;
    size_ =  (int)trainingset.size();
    imgsize_ = row_ * col_;
    computeAverageFace();
    computeMatX();
}

void Training::computeAverageFace()
{
    cv::Mat res = cv::Mat::zeros(row_, col_, CV_16UC1);
    for(auto image:trainingimages_)
    {
        image.convertTo(image, CV_16UC1);
        res += image;
    }
    res /= size_;
    res.convertTo(averageface_, CV_8UC1);
}

void Training::computeMatX()
{
    for(auto trainingimg:trainingimages_)
    {
        cv::Mat img = trainingimg.reshape(0, 1);
        cv::Mat mean = averageface_.reshape(0, 1);
        img.convertTo(img, CV_16SC1);
        mean.convertTo(mean, CV_16SC1);
        img -= mean;
        MatX_.push_back(img);
    }
    MatX_ = MatX_.t();
}

// Find the eigenvectors of the XXT , where X is the data set matrix.
void Training::PCA1()
{
    long long t1 = cv::getTickCount();
    cv::Mat MatX;
    MatX_.convertTo(MatX, CV_16SC1);
    cv::Mat MatXT = MatX.t();
    cv::Mat MatXXT = MatX * MatXT;
    cv::eigen(MatXXT, eigenvalue1_, eigenvector1_);
    
    long long  t2 = cv::getTickCount();
    double time = (t2 - t1)/ cv::getTickFrequency();
    
    cout << "PCA1 took "<< time <<" seconds."<< endl;}

// Find the Principle components by SVD.
void Training::PCA2()
{
    long long t1 = cv::getTickCount();
    cv::Mat MatX;
    MatX_.convertTo(MatX, CV_32FC1);
    cv::SVD::compute(MatX, SIGMA_, U_, VT_);
    
    long long  t2 = cv::getTickCount();
    double time = (t2 - t1)/ cv::getTickFrequency();

    cout << "PCA2 took "<< time <<" seconds."<< endl;

}

void Training::PCA3()
{
    long long t1 = cv::getTickCount();
    cv::Mat MatA;
    
    MatX_.convertTo(MatA, CV_32FC1);
    cv::Mat MatAT = MatA.t();
    cv::Mat MatATA = MatAT * MatA;
    cv::eigen(MatATA, eigenvalue3_, eigenvector3_);
    long long  t2 = cv::getTickCount();
    double time = (t2 - t1)/ cv::getTickFrequency();

    cout << "PCA3 took "<< time <<" seconds."<< endl;

}

void Training::ComputeEigenfaces()
{
    for(int i=0; i<size_; ++i)
    {
        cv::Mat eigenface = cv::Mat::zeros(1, imgsize_, CV_32FC1);
        for(int j=0; j<size_; ++j)
        {
            cv::Mat faceimg = MatX_.col(j).t();
            faceimg.convertTo(faceimg, CV_32FC1);
            faceimg = eigenvector3_.at<float>(i,j) * faceimg;
            eigenface += faceimg;
        }
        eigenfaces_.push_back(eigenface);
    }
    eigenfaces_ = eigenfaces_.t();
}

void Training::FindSigEigenfaces(int n)
{
    cv::Mat SortedIdx;
    cv::sortIdx(eigenvalue3_, SortedIdx, CV_SORT_EVERY_COLUMN | CV_SORT_DESCENDING);
    
    //for()
    
}

int main() {

    Training training_;
    
    vector<cv::Mat> TrainingImgs;
    
    vector<string> faces = {"centerlight", "glasses", "happy","leftlight","noglasses","normal","rightlight","sad","sleepy","surprised","wink"
    };
    
    //assume images have the same size
    for(int i=1; i<=15; i++)
    {
        string subject = to_string(i);
        if(i<10)
            subject = "0" + subject;
        string s = "../../data/TrainingSet/subject" + subject;
        for(auto face:faces)
        {
            string imgpath = s + "." + face;
            cv::Mat img = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
            if(!img.empty())
                TrainingImgs.push_back(img);
        }
    }
    
    training_.initialize(TrainingImgs);
    training_.computeAverageFace();
    //cout<<training_.averageface();
//    cv::namedWindow("TEST", cv::WINDOW_AUTOSIZE);
//    cv::imshow("TEST", training_.averageface());
    
    //training_.PCA1(); //PCA1 takes too much time so it is disabled
    training_.PCA2();
    training_.PCA3();
    
    training_.ComputeEigenfaces();

    int n = 10;
    
    training_.FindSigEigenfaces();

    
//    for(int i=0; i<n; ++i)
//    {
//        cv::Mat eigenface = training_.eigenfaces().col(i).clone();
//        eigenface = eigenface.reshape(0, training_.row());
//        cv::namedWindow("TEST"+to_string(i), cv::WINDOW_AUTOSIZE);
//        cv::imshow("TEST"+to_string(i), eigenface);
//    }
    
    cv::waitKey(0);
    return 0;
}
