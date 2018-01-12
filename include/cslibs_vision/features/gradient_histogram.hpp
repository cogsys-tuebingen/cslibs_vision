#ifndef GRADIENT_HISTOGRAM_HPP
#define GRADIENT_HISTOGRAM_HPP

#include <opencv2/core.hpp>

#include "magnitude.hpp"

namespace cslibs_vision {
struct GradientHistogram {
    GradientHistogram() = delete;

    static inline void standard(const cv::Mat &src,
                                const std::array<double, 2> interval,
                                cv::Mat &dst,
                                cv::Mat &mag,
                                const int ksize = 3)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        cv::Mat magnitude;
        cv::Mat dx;
        cv::Mat dy;
        Magnitude::compute(src, magnitude, dx, dy, ksize);

        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();

        dst = cv::Mat(magnitude.rows, magnitude.cols, CV_8UC1, cv::Scalar());
        uchar *dst_ptr = dst.ptr<uchar>();
        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = std::atan2(dy_ptr[i], dx_ptr[i]);
            if(angle < 0.0)
                angle += M_PI;

            if(angle >= interval[0] &&
                    angle <= interval[1])
                dst_ptr[i] = 1;
        }
        magnitude.copyTo(mag, dst);
    }

    static inline void directed(const cv::Mat &src,
                                const std::array<double, 2> interval,
                                cv::Mat &dst,
                                cv::Mat &mag,
                                const int ksize = 3)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        cv::Mat magnitude;
        cv::Mat dx;
        cv::Mat dy;
        Magnitude::compute(src, magnitude, dx, dy, ksize);

        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();

        dst = cv::Mat(magnitude.rows, magnitude.cols, CV_8UC1, cv::Scalar());
        uchar *dst_ptr = dst.ptr<uchar>();
        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = std::atan2(dy_ptr[i], dx_ptr[i]) + M_PI;
            if(angle >= interval[0] &&
                    angle <= interval[1])
                dst_ptr[i] = 1;
        }
        magnitude.copyTo(mag, dst);
    }

};
}

#endif // GRADIENT_HISTOGRAM_HPP
