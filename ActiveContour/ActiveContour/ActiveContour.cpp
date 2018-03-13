//
//  main.cpp
//  ActiveContour
//
//  Created by TJLin on 2/22/18.
//  Copyright © 2018 TJLin. All rights reserved.
//



#include "ActiveContour.h"
#include "EdgeDetection.h"


bool ActiveContour::isvalid(cv::Point pt)
{
    return pt.x>=0 && pt.y>=0 && pt.x<srcimg_.cols && pt.y<srcimg_.rows;
}

ActiveContour& ActiveContour::neighborhood(int size)
{
    neighborhood_ = size;
    comp_neiborhood_delta();
    return *this;
}

double ActiveContour::MeanPtDist()
{
    double sum = 0;
    int n = dist_.size();
    for(auto& dist : dist_)
        sum += dist;

    return sum/n;
}

void ActiveContour::comp_neiborhood_delta()
{
    drow_.clear();
    dcol_.clear();
    int range = neighborhood_/2;

    for(int i = -range; i<=range; i++)
    {
        for(int j = -range; j<=range; j++)
        {
            drow_.push_back(i);
            dcol_.push_back(j);
        }
    }
}

void ActiveContour::interpolate(double d)
{
    if(d > 5)
    {
        int n = d / 5;
        cv::Point end = points_.back();
        points_.pop_back();
        cv::Point start = points_.back();
        for(int i=1;i<=n;i++)
        {
            double deltax = ((double)end.x - (double)start.x)/(n+1)* i;
            double deltay = ((double)end.y - (double)start.y)/(n+1)* i;
            double x = start.x + deltax;
            double y = start.y + deltay;
            
            cv::Point intP(x,y);
            points_.push_back(intP);
        }
        points_.push_back(end);
    }
}


void ActiveContour::interpolate()
{
    cv::Point start = points_.back();
    cv::Point end = points_.front();
    double dist = cv::norm(end - start);
    if(dist > 5)
    {
        int n = dist / 5;
        for(int i=1;i<=n;i++)
        {
            double deltax = ((double)end.x - (double)start.x)/(n+1)* i;
            double deltay = ((double)end.y - (double)start.y)/(n+1)* i;
            double x = start.x + deltax;
            double y = start.y + deltay;
            
            cv::Point intP(x,y);
            points_.push_back(intP);
        }
    }

    
}

void ActiveContour::Initialize(double alpha, double beta, double gamma)
{
    size_t n = points_.size();
    vector<double> curv(n, 0);
    curv_ = curv;
    dist_.clear();
    para_.clear();
    corners_.clear();
    double dis_i;
    for(int i = 0; i<n;i++)
    {
        dis_i = (i==0)? cv::norm(points_[0] - points_[n-1]):cv::norm(points_[i] - points_[i-1]);
        dist_.push_back(dis_i);
        
        vector<double> para = {alpha, beta, gamma};
        para_.push_back(para);
    }

}

void ActiveContour::onMouse(int evt, int x, int y, int flags, void* param)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        ActiveContour *pThis = (ActiveContour*) param;
        cv::Point newPoint = cv::Point(x,y);
        if(pThis->points().size())
        {
            cv::Point prvPoint = pThis->points().back();
            pThis->points().push_back(newPoint);
            double dist = cv::norm(newPoint - prvPoint);
            pThis->interpolate(dist);
        }else
        {
            pThis->points().push_back(newPoint);
        }
        cout << "position: "<< x << ", " << y << endl;
        pThis->clicks()++;
    }
}

double ActiveContour::compute_Econt(cv::Point P_j, int i, double mean_dis)
{
    size_t n = points_.size();
    double dist;
    
    cv::Point prev = (i == 0) ? points_[n-1] : points_[i-1];
    dist = norm(P_j - prev);
    
    //cout<< dist<<endl;
    
    return abs(mean_dis - dist);
}

double ActiveContour::compute_Ecurv(cv::Point P_j, int i)
{
    size_t n = points_.size();
    double x_comp, y_comp;
    cv::Point prev = (i == 0) ? points_[n-1] : points_[i-1];
    cv::Point next = points_[(i+1)%n];
    
    x_comp = (prev.x - 2 * P_j.x + next.x);
    y_comp = (prev.y - 2 * P_j.y + next.y);

    return x_comp * x_comp + y_comp * y_comp;
}

double ActiveContour::curvature(int i)
{
    size_t n = points_.size();
    double delta_xi, delta_yi, delta_xi1, delta_yi1, delta_si, delta_si1, x_comp, y_comp;
    cv::Point prev = (i == 0) ? points_[n-1] : points_[i-1];
    cv::Point next = points_[(i+1)%n];

    delta_xi = points_[i].x - prev.x;
    delta_xi1 = next.x - points_[i].x;
    delta_yi = points_[i].y - prev.y;
    delta_yi1 = next.y - points_[i].y;
    delta_si = norm(points_[i] - prev);
    delta_si1 = norm(next - points_[i]);
    x_comp = delta_xi/delta_si - delta_xi1/delta_si1;
    y_comp = delta_yi/delta_si - delta_yi1/delta_si1;
    
    return x_comp * x_comp + y_comp * y_comp;
}

double ActiveContour::compute_Eimg(cv::Point P_j, double maxEStr, double minEStr)
{
    double EStr_j = EStr_.at<float>(P_j);
    if( maxEStr - minEStr <5 )
        minEStr = maxEStr - 5;

    return (minEStr-EStr_j)/(maxEStr-minEStr);
}



void ActiveContour::Active_Contour()
{
    size_t n = points_.size();

    double ptsthres = thres_[2] * (double)n;
    double mean_dis = MeanPtDist();
    int ptsmoved;
    int iteration = 0;
    do{
        ptsmoved = 0;
        for(int i = 0; i<=n; i++)
        {
            int index = i%n;

            cv::Point cur, j_min, P_j;
            cur = j_min = points_[index];
            
            double E_j, Econt, Ecurv, Eimg;
            double E_min = std::numeric_limits<double>::max();
            
            double max_Econt, max_Ecurv, maxEStr, minEStr;
            max_Econt = max_Ecurv = maxEStr = std::numeric_limits<double>::min();
            minEStr = std::numeric_limits<double>::max();
            
            
            for(int j = 0; j < neighborhood_ * neighborhood_; j++)
            {
                cv::Point P_j;
                P_j.x = cur.x + dcol_[j];
                P_j.y = cur.y + drow_[j];

                if (isvalid(P_j))
                {
                    Econt = compute_Econt(P_j, i, mean_dis);
                    Ecurv = compute_Ecurv(P_j, i);
                    
                    if(Econt > max_Econt)
                        max_Econt = Econt;
                    
                    if(Ecurv > max_Ecurv)
                        max_Ecurv = Ecurv;
                    
                    if(EStr_.at<float>(P_j) < minEStr)
                        minEStr = EStr_.at<float>(P_j);
                    
                    if(EStr_.at<float>(P_j) > maxEStr)
                        maxEStr = EStr_.at<float>(P_j);
                }

            }
            double E_center = 0;
            for(int j = 0; j < neighborhood_ * neighborhood_; j++)
            {
                cv::Point P_j;
                P_j.x = cur.x + dcol_[j];
                P_j.y = cur.y + drow_[j];
                
                if(isvalid(P_j))
                {
                    Econt = compute_Econt(P_j, i, mean_dis);
                    Ecurv = compute_Ecurv(P_j, i);
                    Eimg = compute_Eimg(P_j, maxEStr, minEStr);
                    
                    E_j = para_[index][0] * Econt/max_Econt + para_[index][1] * Ecurv/max_Ecurv + para_[index][2] * Eimg;

                    if (E_j < E_min)
                    {
                        E_min = E_j;
                        j_min = P_j;
                    }
                }
            }

            points_[index] = j_min;
            if(j_min != cur)
                ptsmoved++;
        }
        
        for (int i=0; i<n; i++)
        {
            double c_i = curvature(i);
            curv_[i] = c_i;
        }
        
        for(int i=0; i<n; i++)
        {
            double prev = (i==0) ? curv_[n-1] : curv_[i-1];
            double next = curv_[(i+1)%n];

            if(curv_[i] > prev && curv_[i] > next && curv_[i] > thres_[0] && EStr_.at<float>(points_[i]) > thres_[1] )
            {
                para_[i][1] = 0;
                auto it = std::find_if(corners_.begin(), corners_.end(), [&](cv::Point a){return points_[i] == a;});
                if(it == corners_.end())
                    corners_.push_back(points_[i]);
            }
        }
        
//        if(iteration == 10)
//        {
//            for (auto it = points_.begin(); it != points_.end(); ++it)
//                cv::circle(display_,*it,1,cv::Scalar(255,213,45),1);
//            cv::polylines(display_, points_, 1, cv::Scalar(255,213,45));
//        }
//
//        if(iteration == 50)
//        {
//            for (auto it = points_.begin(); it != points_.end(); ++it)
//                cv::circle(display_,*it,1,cv::Scalar(87,255,45),1);
//            cv::polylines(display_, points_, 1, cv::Scalar(87,255,45));
//        }
        
        //cout<<"Number of Pts: " << n <<", Pts Moved: " <<ptsmoved <<", Iteration: " << ++iteration <<endl;
        ++iteration;
        mean_dis = MeanPtDist();
    }while(iteration < 1000 && ptsmoved > ptsthres);
    cout<<"Iteration ends, number of Iter = " << iteration << "Number of Pts: " << n << " number of corners = " << to_string(corners_.size()) << endl;
}

int main(int argc, const char * argv[])
{
    vector<cv::Mat> images;
    vector<cv::Mat> colors;
    //part 2 images
//    for(int i=1; i<=8; i++)
//    {
//        string s = "../../src/image" + to_string(i) + ".jpg";
//        images.push_back(cv::imread(s, CV_LOAD_IMAGE_GRAYSCALE));
//        colors.push_back(cv::imread(s, CV_LOAD_IMAGE_COLOR));
//    }
    for(int i=0; i<=19; i++)
    {
        int angle = i*15;
        string s_angle = (angle<100)?("0" + to_string(angle)):to_string(angle);
        s_angle = (angle == 0)?"000":s_angle;
        string s = "../../src/Sequence1/deg" + s_angle + ".jpg";
        images.push_back(cv::imread(s, CV_LOAD_IMAGE_GRAYSCALE));
        colors.push_back(cv::imread(s, CV_LOAD_IMAGE_COLOR));
    }

    int imgno = 0;
    
    vector<int> para = {CV_IMWRITE_JPEG_QUALITY, 100};
    vector<cv::Point> points;
    cv::Mat src = images[imgno].clone();
    cv::Mat color = colors[imgno].clone();
    
    ActiveContour AC(src);
    
    //parameters, change here
    double sigma = 3.0;
    int gaus_size = 5;
    int neighbor_size = 5;
    double alpha = 1, beta = 1, gamma = 1.2;
    double thres1 = 0.5, thres2 = 50, thres3 = 0.1;

    cv::Mat src_f = Mat::zeros( src.rows, src.cols, CV_32F );
    cv::Mat XGrad = Mat::zeros( src.rows, src.cols, CV_32F );
    cv::Mat YGrad = Mat::zeros( src.rows, src.cols, CV_32F );
    cv::Mat EStr = Mat::zeros( src.rows, src.cols, CV_32F );
  
    src.convertTo(src_f, CV_32F);
    src_f = EdgeDetection::Gaussian_Filter(src_f, gaus_size, sigma);
    XGrad = EdgeDetection::GradComponentX(src_f);
    YGrad = EdgeDetection::GradComponentY(src_f);
    EStr = EdgeDetection::EdgeStr(XGrad, YGrad);

    cv::imwrite("../../src/image_es.jpg", EStr, para);
    AC.EdgeStr(EStr);

    cv::namedWindow("TEST", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("TEST", &ActiveContour::onMouse, &AC);

    
    cout << "Press q to continue..." << endl;
    
    while(1)
    {
        AC.display(color.clone());
        for (auto it = AC.points().begin(); it != AC.points().end(); ++it)
            cv::circle(AC.display(),*it,1,cv::Scalar(255,255,255),1);
        
        cv::polylines(AC.display(), AC.points(), 1, cv::Scalar(255,255,255));

        cv::imshow("TEST", AC.display());
        
        if ((cv::waitKey(1) & 0xFF) == 'q')
        {
            if(AC.clicks()<3){
                cout << "Not enough points, please input more points."<<endl;
                cout << "Press q to continue..." << endl;
            }else{
                AC.interpolate();
                for (auto it = AC.points().begin(); it != AC.points().end(); ++it)
                    cv::circle(AC.display(),*it,1,cv::Scalar(255,255,255),1);
                cv::polylines(AC.display(), AC.points(), 1, cv::Scalar(255,255,255));
                break;
            }
        }
    }
    
    cv::imshow("TEST", AC.display());
    //add edge strength/gradient magnitude to AC
    vector<cv::Point> temp;

    temp = AC.points();

    AC.neighborhood(neighbor_size);
    AC.Initialize(alpha, beta, gamma);
    vector<double> thres = {thres1,thres2,thres3};
    AC.thres(thres);
    
    AC.Active_Contour();
    
    for (auto it = AC.points().begin(); it != AC.points().end(); ++it)
        cv::circle(AC.display(),*it,1,cv::Scalar(0,255,255),1);
//    cv::polylines(AC.display(), AC.points(), 1, cv::Scalar(0,255,255));
//    for (auto it = AC.corners().begin(); it != AC.corners().end(); ++it)
//        cv::circle(AC.display(),*it,1,cv::Scalar(0,0,255),1);
    

    cv::imwrite("../../src/Part3/S1/S1_deg000.jpg", AC.display(), para);
    cv::imshow("TEST", AC.display());
    
 
    for(int i= 1; i<=19;i++)
    {
        int angle = i * 15;
        string s_angle = (angle<100)?("0" + to_string(angle)):to_string(angle);

        src = images[i].clone();
        color = colors[i].clone();
        AC.srcimg(src);
        AC.display(color.clone());
        src.convertTo(src_f, CV_32F);
        src_f = EdgeDetection::Gaussian_Filter(src_f, gaus_size, sigma);
        XGrad = EdgeDetection::GradComponentX(src_f);
        YGrad = EdgeDetection::GradComponentY(src_f);
        EStr = EdgeDetection::EdgeStr(XGrad, YGrad);

        //cv::imwrite("../../src/image1_es.jpg", EStr, para);
        AC.EdgeStr(EStr);

        AC.neighborhood(neighbor_size);
        AC.Initialize(alpha, beta, gamma);
        vector<double> thres = {thres1,thres2,thres3};
        AC.thres(thres);

        AC.Active_Contour();

        for (auto it = AC.points().begin(); it != AC.points().end(); ++it)
            cv::circle(AC.display(),*it,1,cv::Scalar(0,255,255),1);
        //cv::polylines(AC.display(), AC.points(), 1, cv::Scalar(0,255,255));
//        for (auto it = AC.corners().begin(); it != AC.corners().end(); ++it)
//            cv::circle(AC.display(),*it,1,cv::Scalar(0,0,255),1);

        cv::imwrite("../../src/Part3/S1/S1_deg" + s_angle + ".jpg", AC.display(), para);
        cv::imshow("TEST" + to_string(i), AC.display());
    }
    
    cv::waitKey(0);
    
    return 0;
}
