#include <opencv2/opencv.hpp>
#include <utils_vision/utils/color_functions.hpp>
#include <utils_vision/utils/stereo_parameters.hpp>

using namespace utils_vision;

#define INPUT_ERROR std::cerr << "utils_vision_test_stereo <video_file> <stereo_parameters>" << std::endl

inline void depthColor(const cv::Mat &src, cv::Mat &dst)
{
    assert(src.type() == CV_32FC3);

    dst = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar());

    cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>();
    const cv::Point3f *src_ptr = src.ptr<cv::Point3f>();

    cv::Mat distances(src.rows, src.cols, CV_32FC1, cv::Scalar());
    float *distances_ptr = distances.ptr<float>();

    float max = std::numeric_limits<float>::min();
    for(unsigned int i = 0 ; i < src.rows ; ++i) {
        for(unsigned int j = 0 ; j < src.cols ; ++j) {
            const cv::Point3f &pt = src_ptr[i * src.cols + j];
            float dist = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
            if(dist == std::numeric_limits<float>::infinity()) {
                dist = 0;
            }

            distances_ptr[i * src.cols + j] = dist;
            if(dist > max && dist < std::numeric_limits<float>::infinity() && dist < 10000)
                max = dist;
        }
    }

    for(unsigned int i = 0; i < src.rows ; ++i) {
        for(unsigned int j = 0 ; j < src.cols ; ++j) {
            dst_ptr[i * src.cols + j] = utils_vision::color::bezierColor<cv::Vec3b>(distances_ptr[i * src.cols + j] / max);
        }
    }
}


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
    cv::Mat points_3D;
    cv::Mat points_3D_vis;
    while(vid.grab()) {
        cv::Mat img;
        vid >> img;

        if(img.type() == CV_8UC3)
            cv::cvtColor(img, img, CV_BGR2GRAY);

        left  = img.colRange(0, cvRound(img.cols / 2));
        right = img.colRange(cvRound(img.cols / 2), img.cols);

        sbm(left, right, disparity, CV_32F);
        cv::reprojectImageTo3D(disparity,points_3D,p.Q);

        double min, max;
        cv::minMaxLoc(disparity, &min, &max);
        cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
        disparity.convertTo( disparity_vis, CV_8UC1, 255.0 );

        depthColor(points_3D, points_3D_vis);

        cv::imshow("video left", left);
        cv::imshow("video right", right);
        cv::imshow("disparity", disparity_vis);
        cv::imshow("depth", points_3D_vis);

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
    cv::Mat points_3D;
    cv::Mat points_3D_vis;
    while(vid.grab()) {
        cv::Mat img;
        vid >> img;

        if(img.type() == CV_8UC3)
            cv::cvtColor(img, img, CV_BGR2GRAY);

        left  = img.colRange(0, cvRound(img.cols / 2));
        right = img.colRange(cvRound(img.cols / 2), img.cols);

        sgbm(left, right, disparity);
        cv::reprojectImageTo3D(disparity,points_3D,p.Q);

        double min, max;
        cv::minMaxLoc(disparity, &min, &max);
        cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
        disparity.convertTo( disparity_vis, CV_8UC1, 255.0 );

        depthColor(points_3D, points_3D_vis);

        cv::imshow("video left", left);
        cv::imshow("video right", right);
        cv::imshow("disparity", disparity_vis);
        cv::imshow("depth", points_3D_vis);

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

