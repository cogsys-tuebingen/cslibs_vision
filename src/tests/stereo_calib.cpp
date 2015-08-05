#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 3) {
        cerr << "utils_vision_stereo_calib <dataset file> <out path>" << endl;
        return 1;
    }

    std::string path_dataset = argv[1];
    std::string path_out = argv[2];
    cv::FileStorage ifs(path_dataset, cv::FileStorage::READ);
    std::vector<std::string> left;
    std::vector<std::string> right;
    int board_w;
    int board_h;

    ifs["left"] >> left;
    ifs["right"] >> right;
    ifs["sx"] >> board_w;
    ifs["sy"] >> board_h;
    ifs.release();

    if(left.size() == 0 || right.size() == 0) {
        std::cerr << "Right or left calibration images missing!" << std::endl;
        return 1;
    }
    if(left.size()  != right.size()) {
        std::cerr << "Need same amount of left and right images!" << std::endl;
        return 1;
    }
    ifs.release();

    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;

    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > imagePoints1, imagePoints2;
    vector<Point2f> corners1, corners2;

    vector<Point3f> obj;
    for (int j=0; j<board_n; j++)
    {
        obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
    }

    Mat img1, img2, gray1, gray2;
    int k = 0;
    bool found1 = false, found2 = false;

    for(unsigned int i = 0 ; i < left.size() ; ++i) {
        img1 = cv::imread(left.at(i), cv::IMREAD_COLOR);
        img2 = cv::imread(right.at(i), cv::IMREAD_COLOR);
        cvtColor(img1, gray1, CV_BGR2GRAY);
        cvtColor(img2, gray2, CV_BGR2GRAY);

        found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1)
        {
            cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray1, board_sz, corners1, found1);
        }

        if (found2)
        {
            cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray2, board_sz, corners2, found2);
        }

        imshow("image1", gray1);
        imshow("image2", gray2);

        k = waitKey(10);
        if (k == 27)
        {
            break;
        }
        if (found1 !=0 && found2 != 0)
        {
//            k = waitKey(0);
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            printf ("Corners stored\n");
        }
    }
    destroyAllWindows();
    printf("Starting Calibration\n");
    Mat CM1 = Mat(3, 3, CV_64FC1);
    Mat CM2 = Mat(3, 3, CV_64FC1);
    Mat D1, D2;
    Mat R, T, E, F;

    stereoCalibrate(object_points, imagePoints1, imagePoints2,
                    CM1, D1, CM2, D2, img1.size(), R, T, E, F,
                    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-5),
                    CV_CALIB_ZERO_TANGENT_DIST);

    FileStorage fso(path_out, FileStorage::WRITE);
    fso << "CM1" << CM1;
    fso << "CM2" << CM2;
    fso << "D1" << D1;
    fso << "D2" << D2;
    fso << "R" << R;
    fso << "T" << T;
    fso << "E" << E;
    fso << "F" << F;

    printf("Done Calibration\n");

    printf("Starting Rectification\n");

    Mat R1, R2, P1, P2, Q;
    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
    fso << "R1" << R1;
    fso << "R2" << R2;
    fso << "P1" << P1;
    fso << "P2" << P2;
    fso << "Q" << Q;
    fso.release();

    printf("Done Rectification\n");

    printf("Applying Undistort\n");

    Mat map1x, map1y, map2x, map2y;
    Mat imgU1, imgU2;

    initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

    printf("Undistort complete\n");


    int pos = 0;
    while(1)
    {
        img1 = cv::imread(left.at(pos), cv::IMREAD_COLOR);
        img2 = cv::imread(right.at(pos), cv::IMREAD_COLOR);

        remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

        imshow("image1", imgU1);
        imshow("image2", imgU2);

        k = waitKey(5);

        if(k==27)
        {
            break;
        }

        ++pos;
        if(pos == left.size())
            pos = 0;
    }

    return(0);
}

