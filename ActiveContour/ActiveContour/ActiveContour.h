//
//  ActiveContour.h
//  ActiveContour
//
//  Created by TJLin on 2/23/18.
//  Copyright © 2018 TJLin. All rights reserved.
//

#ifndef ActiveContour_h
#define ActiveContour_h

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
using namespace std;


class ActiveContour{
public:
    ActiveContour(cv::Mat img):srcimg_(img.clone()), display_(img.clone()){}

    
    cv::Mat& srcimg(){return srcimg_;}
    cv::Mat srcimg() const {return srcimg_;}
    ActiveContour& srcimg(cv::Mat srcimg){srcimg_ = srcimg; return *this;}
    
    cv::Mat& display(){return display_;}
    cv::Mat display() const {return display_;}
    ActiveContour& display(cv::Mat img){display_ = img; return *this;}
    
    cv::Mat& EdgeStr(){return EStr_;}
    cv::Mat EdgeStr() const {return EStr_;}
    ActiveContour& EdgeStr(cv::Mat EStr){EStr_ = EStr; return *this;}
    
    vector<cv::Point>& points() {return points_;}
    vector<cv::Point> points() const {return points_;}
    
    vector<cv::Point>& corners() {return corners_;}
    vector<cv::Point> corners() const {return corners_;}
    
    vector<double>& dist() {return dist_;}
    vector<double> dist() const {return dist_;}
    
    vector<double>& thres() {return thres_;}
    vector<double> thres() const {return thres_;}
    ActiveContour& thres(vector<double> thres){thres_ = thres; return *this;}

    vector<vector<double>>& para() {return para_;}
    vector<vector<double>> para() const {return para_;}
    
    int& clicks(){return click_;}
    int clicks() const{return click_;}

    int& neighborhood(){return neighborhood_;}
    int neighborhood() const{return neighborhood_;}
    ActiveContour& neighborhood(int size);
    
    
    bool isvalid(cv::Point pt);
    double MeanPtDist();
    void comp_neiborhood_delta();
    void interpolate(double d);
    void interpolate();
    static void onMouse(int evt, int x, int y, int flags, void* param);
    void Initialize(double alpha, double beta, double gamma);
    double compute_Econt(cv::Point P_j, int i, double mean_dis);
    double compute_Ecurv(cv::Point P_j, int i);
    double curvature(int i);
    double compute_Eimg(cv::Point P_j, double maxEStr, double minEStr);
    void Active_Contour();
    
    
private:
    cv::Mat srcimg_;
    cv::Mat display_;
    cv::Mat EStr_;
    vector<cv::Point> points_;
    vector<cv::Point> corners_;
    vector<double> dist_;
    vector<double> curv_;
    vector<double> thres_;
    vector<vector<double>> para_;
    int click_;
    
    int neighborhood_;
    vector<int> drow_;
    vector<int> dcol_;
};

#endif /* ActiveContour_h */
