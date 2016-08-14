#ifndef ACF_HPP
#define ACF_HPP

#include <opencv2/opencv.hpp>


namespace cslibs_vision {

/**
 * @brief The ACF - Aggregated Channel Features
 */
class ACF
{
public:
    struct Parameters {
        enum ChannelType {MAGNITUDE, HOG, LUV};
    };

    ACF()
    {
    }

    void compute(const cv::Mat &src,
                 cv::Mat &dst)
    {
        if(src.channels() != 1 && src.channels() != 3) {
            throw std::runtime_error("Channel size must be one or three");
        }

        assert(src.type() == CV_8UC3);
        if(!channels.empty()) {
            channels.clear();
        }

        /// preparation
        cv::Mat kernel = impl::createKernel2D();
        cv::Mat src_as_float;
        src.convertTo(src_as_float, CV_32FC3, 1./255.);
        cv::filter2D(src_as_float, src_as_float, CV_32F, kernel);
        /// normalized gradient magnitude
        cv::Mat norm_grad_mag(src.rows, src.cols, CV_32FC1, cv::Scalar());

        cv::Mat gray;
        cv::Mat dx;
        cv::Mat dy;
        cv::cvtColor(src, gray, CV_BGR2GRAY);
        cv::Sobel(gray, dx, CV_32F, 1, 0);
        cv::Sobel(gray, dy, CV_32F, 0, 1);
        cv::magnitude(dx, dy, norm_grad_mag);

        channels.emplace_back(norm_grad_mag);

        /// HOG with 30° bins
        std::vector<cv::Mat> hog_buffer(6, cv::Mat(norm_grad_mag.rows,
                                                   norm_grad_mag.cols,
                                                   CV_32FC1,
                                                   cv::Scalar()));
        const double bin_step = impl::rad(30.0);
        for(int i = 0 ; i < norm_grad_mag.rows ; ++i) {
            for(int j = 0 ; j < norm_grad_mag.cols ; ++j) {
                double angle = atan2(dy.at<float>(i,j), dx.at<float>(i,j));
                if(angle < 0) {
                    angle += M_PI;
                }
                std::size_t index = (int)(angle / bin_step) % 6;
                hog_buffer.at(index).at<float>(i,j) = norm_grad_mag.at<float>(i,j); // * 255.0
            }
        }

        channels.insert(channels.end(), hog_buffer.begin(), hog_buffer.end());

        /// LUV -> it is important to set the destination type otherwise it won' work!
        cv::Mat luv = cv::Mat(src.rows, src.cols, CV_32FC3, cv::Scalar());
        cv::cvtColor(src_as_float, luv, CV_BGR2Luv);
        std::vector<cv::Mat> luv_buffer;
        cv::split(luv, luv_buffer);
        channels.insert(channels.end(), luv_buffer.begin(), luv_buffer.end());

        int size = 0;
        for(cv::Mat &channel : channels) {
            resample(channel, channel);
            cv::filter2D(channel, channel,CV_32F, kernel);
            size += channel.rows * channel.cols;
        }

        dst = cv::Mat(1, size, CV_32FC1, cv::Scalar());
        float *dst_ptr = dst.ptr<float>();
        int pos = 0;
        for(cv::Mat &channel : channels) {
            if(channel.type() != CV_32FC1)
                throw std::runtime_error("Something went just horribly wrong!");

            int channel_size = channel.rows * channel.cols;
            const float *channel_ptr = channel.ptr<float>();
            for(int i = 0 ; i < channel_size ; ++i) {
                dst_ptr[pos] = channel_ptr[i];
                ++pos;
            }
        }
    }

private:
    constexpr static double RESCALE = 0.5;

    inline void resample(const cv::Mat &src,
                         cv::Mat &dst)
    {
        assert(src.type() == CV_32FC1);

        cv::Mat buffer = cv::Mat(src.rows / 4,
                                 src.cols / 4,
                                 CV_32FC1,
                                 cv::Scalar());

        const static int dx[] = {0, 1, 0, 1};
        const static int dy[] = {0, 0, 1, 1};
        const float * src_ptr = src.ptr<float>();
        float * dst_ptr = buffer.ptr<float>();
        int pos = 0;
        for(int i = 0 ; i < buffer.rows ; ++i) {
            for(int j = 0 ; j < buffer.cols; ++j) {
                int pos_src = 4 * i * src.cols + 4 * j;
                for(int k = 0 ; k < 4 ; ++k) {
                    dst_ptr[pos] += src_ptr[pos_src + dx[k] + dy[k] * src.cols];
                }
                ++pos;
            }
        }
        std::swap(buffer, dst);
    }


    std::vector<cv::Mat> channels;

    inline static const cv::Mat createKernel1D()
    {
        cv::Mat kernel(3,3,CV_32FC1, cv::Scalar());
        const float factor = 1.f/4.f;
        float *data = kernel.ptr<float>();
        data[0] = 1.f * factor;
        data[1] = 2.f * factor;
        data[2] = 1.f * factor;

        return kernel;
    }

    inline static const cv::Mat createKernel2D()
    {
        cv::Mat kernel(3,3,CV_32FC1, cv::Scalar());
        const float factor = 1.f/16.f;
        float *data = kernel.ptr<float>();
        data[0] = 1.f * factor;
        data[1] = 2.f * factor;
        data[2] = 1.f * factor;

        data[3] = 2.f * factor;
        data[4] = 4.f * factor;
        data[5] = 2.f * factor;

        data[6] = 1.f * factor;
        data[7] = 2.f * factor;
        data[8] = 1.f * factor;

        return kernel;
    }

    template<typename T>
    inline T deg(const T rad)
    {
        return M_1_PI * rad * 180.0;
    }

    template<typename T>
    inline T rad(const T deg)
    {
        return M_PI * 1. / 180. * deg;
    }

};

}

#endif // ACF_HPP
