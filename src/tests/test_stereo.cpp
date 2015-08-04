#include <opencv2/opencv.hpp>


#define INPUT_ERROR std::cerr << "utils_vision_test_stereo <video_file> <stereo_parameters>" << std::endl

struct StereoParameters {
    double   base_line;
    double   focal_length;
    double   cx;
    double   xs;
    double   cy;
    double   ys;
    cv::Mat  cam;
    cv::Mat  kl;
    cv::Mat  kr;
    cv::Mat  Q;

    StereoParameters() {
        base_line = 2.3997700199999999e-01;
        focal_length = 9.6899969482400002e+02;
        cy = 4.6353710937500000e+02;
        cx = 6.3513903808600003e+02;
        ys = 960;
        xs = 1280;
    }

    void read(const std::string &path)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        fs["base_line"]    >> base_line;
        fs["focal_length"] >> focal_length;
        fs["cy"]           >> cy;
        fs["cx"]           >> cx;
        fs["ys"]           >> ys;
        fs["xs"]           >> xs;
        fs.release();

    }

    void write(const std::string &path)
    {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        fs << "base_line"    << base_line;
        fs << "focal_length" << focal_length;
        fs << "cy"           << cy;
        fs << "cx"           << cx;
        fs << "ys"           << ys;
        fs << "xs"           << xs;
        fs.release();
    }

};

void runSBM(const std::string &path_video)
{
    cv::VideoCapture vid;
    vid.open(path_video);

    cv::StereoBM sbm;
    sbm.init(CV_STEREO_BM_BASIC, 16 * 5, 5);
    sbm.state->SADWindowSize = 9;
    sbm.state->numberOfDisparities = 112;
    sbm.state->preFilterSize = 5;
    sbm.state->preFilterCap = 61;
    sbm.state->minDisparity = -39;
    sbm.state->textureThreshold = 507;
    sbm.state->uniquenessRatio = 0;
    sbm.state->speckleWindowSize = 0;
    sbm.state->speckleRange = 8;
    sbm.state->disp12MaxDiff = 2;

    cv::Mat left;
    cv::Mat right;
    cv::Mat disparity;
    cv::Mat disparity_vis;
    while(vid.grab()) {
        cv::Mat img;
        vid >> img;

        if(img.type() == CV_8UC3)
            cv::cvtColor(img, img, CV_BGR2GRAY);

        left  = img.colRange(0, cvRound(img.cols / 2));
        right = img.colRange(cvRound(img.cols / 2), img.cols);

        sbm(left, right, disparity, CV_32F);


        double min, max;
        cv::minMaxLoc(disparity, &min, &max);
        cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
        disparity.convertTo( disparity_vis, CV_8UC1, 255.0 );

        cv::imshow("video left", left);
        cv::imshow("video right", right);
        cv::imshow("disparity", disparity_vis);

        int key = cv::waitKey(19) & 0xFF;
        if(key == 27)
            break;
    }

    vid.release();
}

void runSGBM(const std::string &path_video)
{
    cv::VideoCapture vid;
    vid.open(path_video);

    cv::StereoSGBM sgbm;
    sgbm.SADWindowSize = 5;
    sgbm.numberOfDisparities = 192;
    sgbm.preFilterCap = 4;
    sgbm.minDisparity = -64;
    sgbm.uniquenessRatio = 1;
    sgbm.speckleWindowSize = 150;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 10;
    sgbm.fullDP = false;
    sgbm.P1 = 600;
    sgbm.P2 = 2400;

    cv::Mat left;
    cv::Mat right;
    cv::Mat disparity;
    cv::Mat disparity_vis;
    while(vid.grab()) {
        cv::Mat img;
        vid >> img;

        if(img.type() == CV_8UC3)
            cv::cvtColor(img, img, CV_BGR2GRAY);

        left  = img.colRange(0, cvRound(img.cols / 2));
        right = img.colRange(cvRound(img.cols / 2), img.cols);

        sgbm(left, right, disparity);


        double min, max;
        cv::minMaxLoc(disparity, &min, &max);
        cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
        disparity.convertTo( disparity_vis, CV_8UC1, 255.0 );

        cv::imshow("video left", left);
        cv::imshow("video right", right);
        cv::imshow("disparity", disparity_vis);

        int key = cv::waitKey(19) & 0xFF;
        if(key == 27)
            break;
    }

    vid.release();
}


int main(int argc, char *argv[])
{
    if(argc < 3) {
        INPUT_ERROR;
        return 1;
    }

    bool calibrate = argc == 5;

    std::string path_video = argv[1];
    std::string path_calib = argv[2];



    StereoParameters p;
    p.read(path_calib);
    if(calibrate) {
        std::string path_right_calib_pts = argv[3];
        std::string path_left_calib_pts = argv[4];
        std::vector<cv::Point3f> obj_pts_right;
        std::vector<cv::Point3f> obj_pts_left;
        std::vector<cv::Point2f> img_pts_right;
        std::vector<cv::Point2f> img_pts_left;

        cv::FileStorage fsl(path_left_calib_pts, cv::FileStorage::READ);
        fsl["obj"] >> obj_pts_left;
        fsl["img"] >> img_pts_left;
        fsl.release();
        cv::FileStorage fsr(path_right_calib_pts, cv::FileStorage::READ);
        fsr["obj"] >> obj_pts_right;
        fsr["img"] >> img_pts_right;
        fsr.release();

        cv::Mat cam1, dist1;
        cv::Mat cam2, dist2;
        cv::Mat R, T, E, F;
        cv::stereoCalibrate(obj_pts_left, img_pts_left, img_pts_right, cam1, dist1, cam2, dist2,
                            cv::Size(p.xs, p.ys),R,T, E, F);

        cv::Mat R1, P1, R2, P2;
        cv::Mat Q;
        cv::stereoRectify(cam1, dist1, cam2, dist2, cv::Size(p.xs, p.ys), R, T, R1, R2, P1, P2, Q);
        p.Q = Q.clone();
        std::cout << Q << std::endl;
    }

    runSBM(path_video);

    runSGBM(path_video);

    return 0;
}

