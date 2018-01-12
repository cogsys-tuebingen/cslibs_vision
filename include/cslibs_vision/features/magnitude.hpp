#ifndef MAGNITUDE_HPP
#define MAGNITUDE_HPP

#include <opencv2/opencv.hpp>

namespace cslibs_vision {
class Magnitude {
public:
    Magnitude() = delete;

    inline static void compute(const cv::Mat &src,
                               cv::Mat &dst,
                               cv::Mat &dx,
                               cv::Mat &dy,
                               const int ksize = 3)
    {
        cv::Sobel(src, dx, CV_32F, 1, 0, ksize);
        cv::Sobel(src, dy, CV_32F, 0, 1, ksize);
        cv::magnitude(dx, dy, dst);

    }

    inline static void compute(const cv::Mat &src,
                               cv::Mat &dst)
    {
        cv::Mat dx;
        cv::Mat dy;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, dst);
    }

    inline static void compute(const cv::Mat &src,
                               cv::Mat &dst,
                               double &max_magnitude)
    {
        compute(src, dst);
        double min_magnitude;
        cv::minMaxLoc(dst, &min_magnitude, &max_magnitude);
    }
};
}


#endif // MAGNITUDE_HPP
