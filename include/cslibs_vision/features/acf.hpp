#ifndef ACF_HPP
#define ACF_HPP

#include <opencv2/opencv.hpp>
#include <type_traits>

#include "hog.hpp"
#include "lbp.hpp"
#include "ltp.hpp"
#include "wld.hpp"
#include "homogenity.hpp"

namespace cslibs_vision {


class ACF {
public:
    ACF() = delete;

    struct Parameters {
        double hog_bin_size;
        bool   normalize_magnitude;

        Parameters() :
            hog_bin_size(rad(30.0)),
            normalize_magnitude(true)
        {
        }
    };

protected:
    inline static void resample(const cv::Mat &src,
                                cv::Mat &dst)
    {
        assert(src.depth() == CV_32F);

        cv::Mat buffer = cv::Mat(src.rows / 4,
                                 src.cols / 4,
                                 src.type(),
                                 cv::Scalar());

        const static int dx[] = {0, 1, 0, 1};
        const static int dy[] = {0, 0, 1, 1};
        const int channels = src.channels();
        const int step_src = src.cols * channels;
        const int step_dst = dst.cols * channels;
        const float * src_ptr = src.ptr<float>();
        float * dst_ptr = buffer.ptr<float>();
        for(int i = 0 ; i < buffer.rows ; ++i) {
            for(int j = 0 ; j < buffer.cols ; ++j) {
                int pos_dst = step_dst * i + channels * j;
                int pos_src = 4 * (i * step_src + j);
                for(int c = 0 ; c < channels; ++c) {
                    for(int d = 0 ; d < 4 ; ++d) {
                        dst_ptr[pos_dst] += src_ptr[pos_src + dy[k] * step_src + dx[k] * channels + c];
                    }
                }
            }
        }
        dst = std::move(buffer);
    }

    //    inline static void resample(const cv::Mat &src,
    //                                cv::Mat &dst)
    //    {
    //        assert(src.type() == CV_32FC1);

    //        cv::Mat buffer = cv::Mat(src.rows / 4,
    //                                 src.cols / 4,
    //                                 CV_32FC1,
    //                                 cv::Scalar());

    //        const static int dx[] = {0, 1, 0, 1};
    //        const static int dy[] = {0, 0, 1, 1};
    //        const float * src_ptr = src.ptr<float>();
    //        float * dst_ptr = buffer.ptr<float>();
    //        int pos = 0;
    //        for(int i = 0 ; i < buffer.rows ; ++i) {
    //            for(int j = 0 ; j < buffer.cols; ++j) {
    //                int pos_src = 4 * i * src.cols + 4 * j;
    //                for(int k = 0 ; k < 4 ; ++k) {
    //                    dst_ptr[pos] += src_ptr[pos_src + dx[k] + dy[k] * src.cols];
    //                }
    //                ++pos;
    //            }
    //        }
    //        dst = std::move(buffer);
    //    }



    /**
     * @brief createKernel1D produces an [1 2 1] kernel normalized by 4.
     * @return
     */
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

    /**
     * @brief createKernel2D produces a [[1 2 1],[2,4,2],[1,2,1]] kernel normalized by 16.
     * @return
     */
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
    inline static T deg(const T rad)
    {
        return M_1_PI * rad * 180.0;
    }

    template<typename T>
    inline static T rad(const T deg)
    {
        return M_PI * 1. / 180. * deg;
    }
};


/**
 * @brief The ACF - Aggregated Channel Features
 */
class ACFStandard : public ACF
{
public:
    ACFStandard() = delete;

    inline static void compute(const cv::Mat &src,
                               const Parameters &params,
                               cv::Mat &dst)
    {
        const std::size_t channels = src.channels();
        cv::Mat src_as_float;
        if(channels == 1) {
            cv::normalize(src, src_as_float, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

        } else if (channels == 3) {
            cv::normalize(src, src_as_float, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
        } else {
            throw std::runtime_error("Channel size must be one or three");
        }

        cv::Mat kernel = createKernel2D();
        /// 1. filter
        cv::filter2D(src_as_float, src_as_float, CV_32F, kernel);
        if(channels == 1) {
            /// 2. calculate magnitude + 3. calculate hog
            cv::Mat magnitude;
            double  max_magnitude;
            std::vector<cv::Mat> hog;
            HOG::standard(src_as_float, params.hog_bin_size, hog, magnitude, max_magnitude);
            if(!params.normalize_magnitude) {
                max_magnitude = 1.0;
            }

            resample(magnitude, magnitude);
            cv::filter2D(magnitude, magnitude, CV_32F, kernel);
            std::size_t feature_size = magnitude.rows * magnitude.cols;

            for(cv::Mat &h : hog) {
                resample(h,h);
                cv::filter2D(h, h, CV_32F, kernel);
                feature_size += h.cols * h.rows;
            }

            /// 4. LUV cannot be added for grayscale
            /// 5. Copy the data to the destination
            dst = cv::Mat(1, feature_size, CV_32FC1, cv::Scalar());
            float *dst_ptr = dst.ptr<float>();
            int    dst_pos = 0;

            const int    magnitude_size = magnitude.rows * magnitude.cols;
            const float *magnitude_ptr = magnitude.ptr<float>();
            for(int i = 0; i < magnitude_size ; ++i, ++dst_pos) {
                dst_ptr[dst_pos] = magnitude_ptr[i] / max_magnitude;
            }

            for(cv::Mat &h : hog) {
                const int    hog_size = h.rows * h.cols;
                const float *hog_ptr = h.ptr<float>();
                for(int i = 0 ; i < hog_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = hog_ptr[i] / max_magnitude;
                }
            }
        } else {
            cv::Mat gray;
            cv::cvtColor(src, gray, CV_BGR2GRAY);

            /// 2. calculate magnitude + 3. calculate hog
            cv::Mat magnitude;
            double  max_magnitude;
            std::vector<cv::Mat> hog;
            HOG::standard(gray, params.hog_bin_size, hog, magnitude, max_magnitude);
            if(!params.normalize_magnitude) {
                max_magnitude = 1.0;
            }

            resample(magnitude, magnitude);
            cv::filter2D(magnitude, magnitude, CV_32F, kernel);
            std::size_t feature_size = magnitude.rows * magnitude.cols;

            for(cv::Mat &h : hog) {
                resample(h,h);
                cv::filter2D(h,h, CV_32F, kernel);
                feature_size += h.cols * h.rows;
            }

            /// 4. LUV -> it is important to set the destination type otherwise it won' work!
            cv::Mat luv = cv::Mat(src.rows, src.cols, CV_32FC3, cv::Scalar());
            cv::cvtColor(src_as_float, luv, CV_BGR2Luv);
            resample(luv, luv);
            cv::filter2D(luv, luv, CV_32F, kernel);
            feature_size += luv.cols * luv.rows * 3; // 3 <=> L * U * V

            dst = cv::Mat(1, feature_size, CV_32FC1, cv::Scalar());
            float *dst_ptr = dst.ptr<float>();
            int    dst_pos = 0;

            const int    magnitude_size = magnitude.rows * magnitude.cols;
            const float *magnitude_ptr = magnitude.ptr<float>();
            for(int i = 0; i < magnitude_size ; ++i, ++dst_pos) {
                dst_ptr[dst_pos] = magnitude_ptr[i] / max_magnitude;
            }

            for(cv::Mat &h : hog) {
                const int    hog_size = h.rows * h.cols;
                const float *hog_ptr = h.ptr<float>();
                for(int i = 0 ; i < hog_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = hog_ptr[i] / max_magnitude;
                }
            }

            const int luv_size = luv.rows * luv.cols * 3;
            const float *luv_ptr = luv.ptr<float>();
            for(int i = 0 ; i < luv_size ; ++i, ++dst_pos) {
                dst_ptr[dst_pos] = luv_ptr[i];
            }
        }

    }
};

class ACFDynamic : public ACF
{
public:
    ACFDynamic() = delete;

    struct Parameters : public ACF::Parameters {
        enum ChannelType {MAGNITUDE = 1, HOG = 2, LUV = 4, LBP = 8, LTP = 16, WLD = 32, HOMOGENITY = 64};
        ChannelType channel_types;
        Parameters() :
            ACF::Parameters(),
            channel_types(MAGNITUDE + HOG + LUV)
        {
        }
    };


//    inline static void compute(const cv::Mat &src,
//                               const Parameters &params,
//                               cv::Mat &dst)
//    {
//        const std::size_t channels = src.channels();
//        cv::Mat src_as_float;
//        if(channels == 1) {
//            cv::normalize(src, src_as_float, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

//        } else if (channels == 3) {
//            cv::normalize(src, src_as_float, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
//        } else {
//            throw std::runtime_error("Channel size must be one or three");
//        }

//        cv::Mat kernel = createKernel2D();
//        /// 1. filter
//        cv::filter2D(src_as_float, src_as_float, CV_32F, kernel);
//        if(channel == 1) {
//            std::vector<cv::Mat>     channels(channel_count, cv::Mat());
//            std::vector<std::size_t> channel_sizes(channel_count, 0);
//            std::vector<double>      normalization(channel_count, 1.0);
//            std::size_t              pos = 0;

//            if(params.channel_types & Parameters::HOG != 0) {
//                cv::Mat magnitude;
//                double  max_magnitude;
//                std::vector<cv::Mat> hog;
//                HOG::standard(gray, params.hog_bin_size, hog, magnitude, max_magnitude);

//                if(!params.normalize_magnitude) {
//                    max_magnitude = 1.0;
//                }

//                if(params.channel_types & Parameters::MAGNITUDE) {
//                    channels[pos] = std::move(magnitude);
//                    normalization[pos] = max_magnitude;
//                    channel_sizes[pos] = magnitude.cols * magnitude.rows;
//                    ++pos;
//                }
//                for(cv::Mat &h : hog) {
//                    channels[pos] = std::move(h);
//                    normalization[pos] = max_magnitude;
//                    channel_sizes[pos] = h.cols * h.rows;
//                    ++pos;
//                }

//            } else if(params.channel_types & Parameters::MAGNITUDE != 0) {
//                cv::Mat dx;
//                cv::Mat dy;
//                cv::Sobel(src, dx, CV_32F, 1, 0);
//                cv::Sobel(src, dy, CV_32F, 0, 1);
//                cv::magnitude(dx, dy, channels[pos]);
//                cv::Mat &magnitude = channels[pos];
//                channel_sizes[pos] = magnitude.cols * magnitude.rows;
//                if(params.normalize_magnitude) {
//                    double min;
//                    cv::minMaxLoc(magnitude, &min, &normalization[pos]);
//                }
//                ++pos;
//            }
//            if(params.channel_types & Parameters::LBP != 0) {

//            }
//            if(params.channel_types & Parameters::LTP != 0) {

//            }
//            if(params.channel_types & Parameters::WLD != 0) {

//            }
//            if(params.channel_types & Parameters::HOMOGENITY != 0) {

//            }
//        } else {
//            cv::Mat gray;
//            cv::cvtColor(src, gray, CV_BGR2GRAY);
//            if(params.channel_types & Parameters::MAGNITUDE != 0) {

//            }
//            if(params.channel_types & Parameters::HOG != 0) {

//            }
//            if(params.channel_types & Parameters::LBP != 0) {

//            }
//            if(params.channel_types & Parameters::LTP != 0) {

//            }
//            if(params.channel_types & Parameters::WLD != 0) {

//            }
//            if(params.channel_types & Parameters::HOMOGENITY != 0) {

//            }
//        }
//    }

//protected:
//    template<std::size_t N>
//    struct Bits {
//        static_assert(N > 0, "N must be greater 0");
//        inline static std::size_t count(const int n)
//        {
//            return (n & 0x1) + Bits<N-1>::count(n >> 1);
//        }

//    };

//    template<>
//    struct Bits<0> {
//        inline static std::size_t count(const int n)
//        {
//            return n & 0x1;
//        }

//    };


};

}

#endif // ACF_HPP
