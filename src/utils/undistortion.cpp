#include <utils_vision/utils/undistortion.h>
#include <opencv2/opencv.hpp>

using namespace utils_vision;

Undistortion::Undistortion(const cv::Mat &camera_mat, const cv::Mat &distoration_coeffs, const int interpolation) :
    interpolation_method_(interpolation),
    camera_mat_(camera_mat),
    distortion_coeffs_(distoration_coeffs)
{
}

Undistortion::~Undistortion()
{
}

void Undistortion::undistort_nomap(const cv::Mat &src, cv::Mat &dst)
{
    cv::undistort(src, dst, camera_mat_, distortion_coeffs_);
}

void Undistortion::undistort(const cv::Mat &src, cv::Mat &dst)
{
    cv::remap(src, dst, undist_map_x_, undist_map_y_, interpolation_method_);
}

void Undistortion::undistort(const cv::Mat &src, cv::Mat &dst, int margin_x, int margin_y)
{
    cv::Mat tmp;
    cv::Mat img_scaled(src.rows + 2 * margin_y, src.cols + 2 * margin_x, src.type(), cv::Scalar::all(0));
    cv::Mat roi(img_scaled, cv::Rect(margin_x, margin_y, src.cols, src.rows));
    src.copyTo(roi);
    undistort(img_scaled, tmp);
    cv::resize(tmp, tmp, cv::Size(src.cols, src.rows), 0, 0, interpolation_method_);
    tmp.copyTo(dst);
}

void Undistortion::undistort_points_nomap(const cv::Mat &src, cv::Mat &dst)
{
    cv::undistortPoints(src, dst, camera_mat_, distortion_coeffs_);
}

void Undistortion::undistort_points_nomap(const std::vector<cv::Point2d> &src, std::vector<cv::Point2d> &dst)
{
    cv::undistortPoints(src, dst, camera_mat_, distortion_coeffs_);
}

void Undistortion::undistort_points_nomap(const std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst)
{
    cv::undistortPoints(src, dst, camera_mat_, distortion_coeffs_);
}

void Undistortion::reset_map(const cv::Size s, const float camera_offset_x, const float camera_offset_y)
{
    if(undist_img_size_ != s) {
        cv::Mat tmp_cam = camera_mat_.clone();

        tmp_cam.at<float>(0,2) += camera_offset_x;
        tmp_cam.at<float>(1,2) += camera_offset_y;

        cv::initUndistortRectifyMap(tmp_cam, distortion_coeffs_,
                                    orientation_, optimal_camera_mat_, s,
                                    CV_16SC2, undist_map_x_, undist_map_y_);
        undist_img_size_ = s;
    }
}

void Undistortion::reset_interpolation(const int method)
{
    interpolation_method_ = method;
}


