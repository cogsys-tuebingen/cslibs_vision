#include <opencv2/opencv.hpp>


#define INPUT_ERROR std::cerr << "utils_vision_test_stereo <video_file> <stereo_parameters>" << std::endl

struct StereoParameters {
    cv::Mat CM1, CM2;
    cv::Mat D1, D2;
    cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;


    void write(const std::string &calibration) {
        cv::FileStorage fso(calibration, cv::FileStorage::WRITE);
        fso << "CM1" << CM1;
        fso << "CM2" << CM2;
        fso << "D1"  << D1;
        fso << "D2"  << D2;
        fso << "R"   << R;
        fso << "T"   << T;
        fso << "E"   << E;
        fso << "F"   << F;
        fso << "R1"  << R1;
        fso << "R2"  << R2;
        fso << "P1"  << P1;
        fso << "P2"  << P2;
        fso << "Q"   << Q;
        fso.release();
    }
    void read(const std::string &calibration) {
        cv::FileStorage fsi(calibration, cv::FileStorage::READ);
        fsi["CM1"] >> CM1;
        fsi["CM2"] >> CM2;
        fsi["D1"]  >> D1;
        fsi["D2"]  >> D2;
        fsi["R"]   >> R;
        fsi["T"]   >> T;
        fsi["E"]   >> E;
        fsi["F"]   >> F;
        fsi["R1"]  >> R1;
        fsi["R2"]  >> R2;
        fsi["P1"]  >> P1;
        fsi["P2"]  >> P2;
        fsi["Q"]   >> Q;
        fsi.release();
    }
};

void runSBM(const std::string &path_video,
            const StereoParameters &p)
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

void runSGBM(const std::string &path_video,
             const StereoParameters &p)
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

    std::string path_video = argv[1];
    std::string path_calib = argv[2];
    StereoParameters p;
    p.read(path_calib);

    runSBM(path_video, p);
    runSGBM(path_video, p);

    return 0;
}

