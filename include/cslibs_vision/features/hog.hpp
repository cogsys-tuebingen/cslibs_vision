#ifndef HOG_HPP
#define HOG_HPP

#include <opencv2/opencv.hpp>

namespace cslibs_vision {
class HOG {
public:
    HOG() = delete;

    /// -UNDIRECTED------------------------------------------------------------------------------------------ ///
    /// ----------------------------------------------------------------------------------------------------- ///
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

        const std::size_t bins = M_PI / bin_size;
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

            std::size_t index = (int)(angle / bin_size) % bins;
            dst_bins_ptr[i] = index;
            dst_weights_ptr[i] = magnitude_ptr[i]; // *  255
        }
    }

    static inline void standard(const cv::Mat        &src,
                                const double          bin_size,
                                std::vector<cv::Mat> &dst)
    {
        cv::Mat magnitude;
        double  max_magnitude;
        standard(src, bin_size, dst, magnitude, max_magnitude);
    }

    static inline void standard(const cv::Mat        &src,
                                const double          bin_size,
                                std::vector<cv::Mat> &dst,
                                cv::Mat              &magnitude,
                                double               &max_magnitude)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        cv::Mat dx;
        cv::Mat dy;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, magnitude);
        dst.resize(bins, cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar()));
        max_magnitude = std::numeric_limits<double>::lowest();

        const std::size_t bins = M_PI / bin_size;
        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        std::vector<float*> ptr_to_channels;
        for(cv::Mat &channel : bins) {
            ptr_to_channels.emplace_back(channel.ptr<float>());
        }
        float **channels_ptr = ptr_to_channels.data();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]);
            if(angle < 0.0)
                angle += M_PI;

            double m = magnitude_ptr[i];
            if(m > max_magnitude) {
                max_magnitude = m;
            }
            std::size_t index = angle / bin_size % bins;
            channels_ptr[index][i] = m; // * 255
        }
    }

    /// -DIRECTED-------------------------------------------------------------------------------------------- ///
    /// ----------------------------------------------------------------------------------------------------- ///
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

        const std::size_t bins = 2 * M_PI / bin_size;
        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        uchar *dst_bins_ptr = dst_bins.ptr<uchar>();
        float *dst_weights_ptr = dst_weights.ptr<float>();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]) + M_PI;
            std::size_t index = angle / bin_size % bins;
            dst_bins_ptr[i] = index;
            dst_weights_ptr[i] = magnitude_ptr[i]; // *  255
        }
    }

    static inline void directed(const cv::Mat        &src,
                                const double          bin_size,
                                std::vector<cv::Mat> &dst)
    {
        cv::Mat magnitude;
        double max_magnitude;
        standard(src, bin_size, dst, magnitude, max_magnitude);
    }


    static inline void directed(const cv::Mat        &src,
                                const double          bin_size,
                                std::vector<cv::Mat> &dst,
                                cv::Mat              &magnitude,
                                double               &max_magnitude)
    {
        if(src.channels() > 1)
            throw std::runtime_error("Input matrix needs to be single channel!");

        cv::Mat dx;
        cv::Mat dy;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, magnitude);
        dst.resize(channels, cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar()));

        const std::size_t bins = 2 * M_PI / bin_size;
        const float *dx_ptr = dx.ptr<float>();
        const float *dy_ptr = dy.ptr<float>();
        const float *magnitude_ptr = magnitude.ptr<float>();
        std::vector<float*> ptr_to_channels;
        for(cv::Mat &channel : dst) {
            ptr_to_channels.emplace_back(channel.ptr<float>());
        }
        float **channels_ptr = ptr_to_channels.data();

        const int size = magnitude.rows * magnitude.cols;
        for(int i = 0 ; i < size ; ++i) {
            double angle = atan2(dy_ptr[i], dx_ptr[i]) + M_PI;

            double m = magnitude_ptr[i];
            if(m > max_magnitude) {
                max_magnitude = m;
            }

            std::size_t index = angle / bin_size % bins;
            channels_ptr[index][i] = m; // * 255
        }
    }


};
}

#endif // HOG_HPP
