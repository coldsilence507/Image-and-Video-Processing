#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

Mat Mean_Filter(Mat src, Mat res, int n){
    try{
        if(n%2 != 1) throw 1;
    }
    catch(int n){
        cerr<<"Kernel size should not be even.";
    }
    Mat kernel;
    int kernel_size;
    kernel_size = n;
    Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth = -1;
    
    kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/(kernel_size*kernel_size);
    
    filter2D(src, res, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
    return res;
}

double Gaussian(int n, double sigma, int mu = 0)
{
    return 1/sqrt(2*M_PI*sigma)*exp(-pow(n,2)/(2*pow(sigma, 2)));
}

Mat genGaussKernelX(int n, double sigma, int mu = 0){
    try{
        if(n < 5 || n%2 != 1) throw 1;
    }
    catch(int n){
        cerr<<"Kernel size should not be less than 5 or even.";
    }
    Mat GaussX(1, n, CV_64F);
    int mid = n/2;

    for(int i = 0; i<n; i++){
            GaussX.at<double>(0,i) = Gaussian(i-mid, sigma, mu);
    }
    double sum = cv::sum(GaussX)[0];
    GaussX /= sum;
    return GaussX;
}


Mat Gaussian_Filter(Mat src, Mat res, int n, double sigma, double mu = 0){
    Mat kernel;
    int kernel_size;
    kernel_size = n;
    Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth = -1;
    
    Mat GaussX, GaussY;
    GaussX = genGaussKernelX(n, sigma);
    transpose(GaussX, GaussY);
    
    filter2D(src, src, ddepth , GaussX, anchor, delta, BORDER_DEFAULT );
    filter2D(src, res, ddepth , GaussY, anchor, delta, BORDER_DEFAULT );
    
    return res;
}

Mat Median_Filter(Mat src, Mat res, int n)
{
    Mat kernel;

    for(int i = 0; i<src.rows-n; i++){
        for(int j =0; j<src.cols-n;j++)
        {
            
             vector<uchar> array;
            for(int x = i; x<i+n; x++)
                for(int y = j; y<j+n; y++)
                    array.push_back(src.at<uchar>(x,y));
            
            sort(array.begin(), array.end());
            int med = n*n/2;
            int row = i+n/2;
            int col = j+n/2;
            res.at<uchar>(row, col) = array[med];
        }
    }
    return res;
}



/** @function main */
int main()
{
    Mat srcimg1, srcimg2, resimg1, resimg2;
    /// Load an image
    srcimg1 = imread( "../../src/NoisyImage1.jpg",IMREAD_GRAYSCALE );
    srcimg2 = imread( "../../src/NoisyImage2.jpg",IMREAD_GRAYSCALE );
    resimg1 = Mat::zeros( srcimg1.rows, srcimg1.cols, CV_8U );
    resimg2 = Mat::zeros( srcimg2.rows, srcimg2.cols, CV_8U );

    
    if( !srcimg1.data || !srcimg2.data )
    {   cout<< "Failed to load source image." << endl;
        return -1; }
    
    vector<int> para = {CV_IMWRITE_JPEG_QUALITY, 100};

    //choice
    char n;
    cout<<"Please select filter type:" << endl << "1. Mean Filter;" << endl<< "2. Gaussian Filter;";
    cout<< endl <<"3. Median Filter;" << endl << "4.Exit." << endl;
    cin >> n;
    switch(n){
        case '1':{
            int size;
            cout<<"please input the kernel size (int):";
            cin >> size;
            resimg1 = Mean_Filter(srcimg1, resimg1, size);
            resimg2 = Mean_Filter(srcimg2, resimg2, size);
            imwrite("../../src/Mean_NoisyImage1_" + to_string(size) + "*" + to_string(size)  +".jpg", resimg1, para);
            imwrite("../../src/Mean_NoisyImage2_" + to_string(size) + "*" + to_string(size)  +".jpg", resimg2, para);
            break;
        }
        case '2':{
            float sigma;
            int size;
            cout<<"please input the kernel size (int):";
            cin>>size;
            cout<<"please input the sigma value (default kernel size = 5):";
            cin >> sigma;
            
            resimg1 = Gaussian_Filter(srcimg1, resimg1, size, sigma);
            resimg2 = Gaussian_Filter(srcimg2, resimg2, size, sigma);
            
            imwrite("../../src/Gaussian_NoisyImage1_sigma_" + to_string(sigma) +".jpg", resimg1, para);
            imwrite("../../src/Gaussian_NoisyImage2_sigma_" + to_string(sigma) +".jpg", resimg2, para);
            break;
        }
        case '3':{
            int size;
            cout<<"please input the kernel size (int):";
            cin >> size;
            resimg1 = Median_Filter(srcimg1, resimg1, size);
            resimg2 = Median_Filter(srcimg2, resimg2, size);
            imwrite("../../src/Median_NoisyImage1_" + to_string(size) + "*" + to_string(size)  +".jpg", resimg1, para);
            imwrite("../../src/Median_NoisyImage2_" + to_string(size) + "*" + to_string(size)  +".jpg", resimg2, para);
            break;
            break;
        }
        case '4':
            return 0;
        default:
            break;
    }
    
    getchar();
    return 0;
}
