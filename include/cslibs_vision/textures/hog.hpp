#ifndef HOG_HPP
#define HOG_HPP

#include <opencv2/opencv.hpp>

namespace cslibs_vision {
class HOG {
public:
    static inline void standard(const cv::Mat &src,
                                const double   bin_size,
                                cv::Mat       &dst_bins,
                                cv::Mat       &dst_weights)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        cv::Mat dx;
        cv::Mat dy;
        cv::Mat magnitude;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, magnitude);
        dst_bins = cv::Mat(magnitude.rows, magnitude.cols, CV_8UC1, cv::Scalar());
        dst_weights = cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar());

        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        uchar *dst_bins_ptr = dst_bins.ptr<uchar>();
        float *dst_weights_ptr = dst_weights.ptr<float>();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]);
            if(angle < 0.0)
                angle += M_PI;

            dst_bins_ptr[i] = angle / bin_size;
            dst_weights_ptr[i] = magnitude_ptr[i]; // *  255
        }
    }

    static inline void standard(const cv::Mat        &src,
                                const double          bin_size,
                                std::vector<cv::Mat> &dst)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        const std::size_t channels = M_PI / bin_size;
        cv::Mat dx;
        cv::Mat dy;
        cv::Mat magnitude;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, magnitude);
        dst.resize(channels, cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar()));

        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        std::vector<float*> ptr_to_channels;
        for(cv::Mat &channel : channels) {
            ptr_to_channels.emplace_back(channel.ptr<float>());
        }
        float **channels_ptr = ptr_to_channels.data();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]);
            if(angle < 0.0)
                angle += M_PI;

            channels_ptr[angle/bin_size][i] = magnitude_ptr[i]; // * 255
        }
    }

    static inline void directed(const cv::Mat &src,
                                const double   bin_size,
                                cv::Mat       &dst_bins,
                                cv::Mat       &dst_weights)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");


        cv::Mat dx;
        cv::Mat dy;
        cv::Mat magnitude;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, magnitude);
        dst_bins = cv::Mat(magnitude.rows, magnitude.cols, CV_8UC1, cv::Scalar());
        dst_weights = cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar());

        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        uchar *dst_bins_ptr = dst_bins.ptr<uchar>();
        float *dst_weights_ptr = dst_weights.ptr<float>();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]) + M_PI;
            dst_bins_ptr[i] = angle / bin_size;
            dst_weights_ptr[i] = magnitude_ptr[i]; // *  255
        }
    }

    static inline void directed(const cv::Mat        &src,
                                const double          bin_size,
                                std::vector<cv::Mat> &dst)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        const std::size_t channels = 2 * M_PI / bin_size;
        cv::Mat dx;
        cv::Mat dy;
        cv::Mat magnitude;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, magnitude);
        dst.resize(channels, cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar()));

        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        std::vector<float*> ptr_to_channels;
        for(cv::Mat &channel : channels) {
            ptr_to_channels.emplace_back(channel.ptr<float>());
        }
        float **channels_ptr = ptr_to_channels.data();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]);
            angle += M_PI;

            channels_ptr[angle/bin_size][i] = magnitude_ptr[i]; // * 255
        }
    }


};
}

#endif // HOG_HPP
