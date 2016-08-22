#ifndef ACF_HPP
#define ACF_HPP

#include <opencv2/opencv.hpp>
#include <type_traits>

#include "hog.hpp"
#include "lbp.hpp"
#include "ltp.hpp"
#include "wld.hpp"
#include "homogenity.hpp"
#include "magnitude.hpp"

namespace cslibs_vision {

namespace impl {
template<std::size_t N>
struct Bits {
    static_assert(N > 0, "N must be greater 0");
    inline static std::size_t count(const int n)
    {
        return (n & 0x1) + Bits<N-1>::count(n >> 1);
    }

};

template<>
struct Bits<0> {
    inline static std::size_t count(const int n)
    {
        return n & 0x1;
    }

};
}


class ACF {
public:
    ACF() = delete;

    struct Parameters {
        enum KernelType {NONE, KERNEL_1D, KERNEL_2D};

        double     hog_bin_size;
        bool       hog_directed ;
        bool       normalize_magnitude;
        KernelType kernel_type;

        Parameters() :
            hog_bin_size(rad(30.0)),
            hog_directed(false),
            normalize_magnitude(true),
            kernel_type(KERNEL_2D)
        {
        }
    };

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
protected:
    struct ResamplingBlockSize {
        const static int width = 4;
        const static int height = 4;
    };


    template<typename T>
    inline static void resample(const cv::Mat &src,
                                cv::Mat &dst)
    {
        cv::Mat buffer = cv::Mat(src.rows / ResamplingBlockSize::height,
                                 src.cols / ResamplingBlockSize::width,
                                 CV_32FC(src.channels()),
                                 cv::Scalar());

        assert(src.depth() == CV_32F);

        const static int dx[] = {0, 1, 0, 1};
        const static int dy[] = {0, 0, 1, 1};
        const int channels = src.channels();
        const int step_src = src.cols * channels;
        const int step_buffer = buffer.cols * channels;
        const T * src_ptr = src.ptr<T>();
        float * buffer_ptr = buffer.ptr<float>();
        for(int i = 0 ; i < buffer.rows ; ++i) {
            for(int j = 0 ; j < buffer.cols ; ++j) {
                int pos_buffer = step_buffer * i + channels * j;
                int pos_src = 4 * (i * step_src + j);
                for(int c = 0 ; c < channels; ++c) {
                    for(int d = 0 ; d < 4 ; ++d) {
                        buffer_ptr[pos_buffer] += (float) src_ptr[pos_src + dy[d] * step_src + dx[d] * channels + c];
                    }
                }
            }
        }
        dst = buffer;
    }

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
        if(channels == 1 || channels == 3) {
            cv::normalize(src, src_as_float, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
        } else {
            throw std::runtime_error("Channel size must be one or three");
        }

        cv::Mat kernel;
        switch(params.kernel_type) {
        case Parameters::KERNEL_1D:
            kernel = createKernel1D();
            break;
        case Parameters::KERNEL_2D:
            kernel = createKernel2D();
            break;
        default:
            throw std::runtime_error("Unknown kernel type!");
        }

        /// 1. filter
        if(params.kernel_type != Parameters::NONE)
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

            resample<float>(magnitude, magnitude);
            if(params.kernel_type != Parameters::NONE)
                cv::filter2D(magnitude, magnitude, CV_32F, kernel);
            std::size_t feature_size = magnitude.rows * magnitude.cols;

            for(cv::Mat &h : hog) {
                resample<float>(h,h);
                if(params.kernel_type != Parameters::NONE)
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
            cv::Mat gray_as_float(src_as_float.rows, src_as_float.rows, CV_32FC1, cv::Scalar());
            cv::cvtColor(src_as_float, gray_as_float, CV_BGR2GRAY);

            /// 2. calculate magnitude + 3. calculate hog
            cv::Mat magnitude;
            double  max_magnitude;
            std::vector<cv::Mat> hog;
            HOG::standard(gray_as_float, params.hog_bin_size, hog, magnitude, max_magnitude);
            if(!params.normalize_magnitude) {
                max_magnitude = 1.0;
            }

            resample<float>(magnitude, magnitude);
            if(params.kernel_type != Parameters::NONE)
                cv::filter2D(magnitude, magnitude, CV_32F, kernel);
            std::size_t feature_size = magnitude.rows * magnitude.cols;

            for(cv::Mat &h : hog) {
                resample<float>(h,h);
                if(params.kernel_type != Parameters::NONE)
                    cv::filter2D(h,h, CV_32F, kernel);
                feature_size += h.cols * h.rows;
            }

            /// 4. LUV -> it is important to set the destination type otherwise it won' work!
            cv::Mat luv = cv::Mat(src.rows, src.cols, CV_32FC3, cv::Scalar());
            cv::cvtColor(src_as_float, luv, CV_BGR2Luv);
            resample<float>(luv, luv);
            if(params.kernel_type != Parameters::NONE)
                cv::filter2D(luv, luv, CV_32F, kernel);
            feature_size += luv.cols * luv.rows * 3; // 3 <=> L * U * V

            dst = cv::Mat(1, feature_size, CV_32FC1, cv::Scalar());
            float *dst_ptr = dst.ptr<float>();
            int    dst_pos = 0;

            const int    magnitude_size = magnitude.rows * magnitude.cols;
            const float *magnitude_ptr  = magnitude.ptr<float>();
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

            const int luv_channels = luv.channels();
            const float *luv_ptr = luv.ptr<float>();
            const int luv_step = luv_channels * luv.cols;
            for(int c = 0 ; c < luv_channels ; ++c) {
                for(int i = 0 ; i < luv.rows ; ++i) {
                    for(int j = 0 ; j < luv.cols ; ++j) {
                        dst_ptr[dst_pos] = luv_ptr[i * luv_step + j * luv_channels + c];
                        ++dst_pos;
                    }
                }
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

        int             channel_types;
        bool            normalize_patterns;
        double          k;

        Parameters() :
            ACF::Parameters(),
            channel_types(MAGNITUDE | HOG | LUV),
            normalize_patterns(false),
            k(0)
        {
        }

        inline void setChannel(const ChannelType c)
        {
            channel_types |= c;
        }

        inline void unsetChannel(const ChannelType c)
        {
            channel_types &= ~c;
        }

        inline std::size_t featureSize(const int src_rows,
                                       const int src_cols) const
        {

            std::size_t size = 0;
            if(has(MAGNITUDE)) {
                size += src_rows / ResamplingBlockSize::height *
                        src_cols / ResamplingBlockSize::width;
            }
            if(has(HOG)) {
                size +=  src_rows / ResamplingBlockSize::height *
                        src_cols / ResamplingBlockSize::width *
                        cslibs_vision::HOG::standarBins(hog_bin_size);
            }
            if(has(LUV)) {
                size += src_rows / ResamplingBlockSize::height *
                        src_cols / ResamplingBlockSize::width *
                        3;  /// L U V
            }
            if(has(LBP)) {
                assert(src_rows > 2);
                assert(src_cols > 2);
                size += cslibs_vision::LBP::standardRows(src_rows) / ResamplingBlockSize::height *
                        cslibs_vision::LBP::standardCols(src_cols) / ResamplingBlockSize::width;
            }
            if(has(LTP)) {
                assert(src_rows > 2);
                assert(src_cols > 2);
                size += cslibs_vision::LTP::standardRows(src_rows) / ResamplingBlockSize::height *
                        cslibs_vision::LTP::standardCols(src_cols) / ResamplingBlockSize::width *
                        2; /// upper and lower ternary channel
            }
            if(has(WLD)) {
                assert(src_rows > 2);
                assert(src_cols > 2);
                size += cslibs_vision::WLD::standardRows(src_rows) / ResamplingBlockSize::height *
                        cslibs_vision::WLD::standardCols(src_cols) / ResamplingBlockSize::width;
            }
            if(has(HOMOGENITY)) {
                assert(src_rows > 2);
                assert(src_cols > 2);
                size += cslibs_vision::Homogenity::standardRows(src_rows) / ResamplingBlockSize::height *
                        cslibs_vision::Homogenity::standardCols(src_cols) / ResamplingBlockSize::width;
            }
            return size;
        }

        inline bool has(const ChannelType type) const
        {
            return (channel_types & type) != 0x0;
        }
    };

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
            //// 1 Channel
            if(params.has(Parameters::LUV))
                throw std::runtime_error("Cannot use channel type 'LUV' with mono channel images!");

            const std::size_t feature_size = params.featureSize(src.rows, src.cols);
            dst = cv::Mat(1, feature_size, CV_32FC1, cv::Scalar());
            float      *dst_ptr = dst.ptr<float>();
            std::size_t dst_pos = 0;

            if(params.has(Parameters::HOG)) {
                cv::Mat magnitude;
                double  max_magnitude;
                std::vector<cv::Mat> hog;
                HOG::standard(src_as_float, params.hog_bin_size, hog, magnitude, max_magnitude);
                if(!params.normalize_magnitude) {
                    max_magnitude = 1.0;
                }

                if(params.has(Parameters::MAGNITUDE)) {
                    resample<float>(magnitude, magnitude);
                    cv::filter2D(magnitude, magnitude, CV_32F, kernel);

                    const int    magnitude_size = magnitude.rows * magnitude.cols;
                    const float *magnitude_ptr  = magnitude.ptr<float>();
                    for(int i = 0 ; i < magnitude_size ; ++i, ++dst_pos) {
                        dst_ptr[dst_pos] = magnitude_ptr[i] / max_magnitude;
                    }
                }
                for(cv::Mat &h : hog) {
                    resample<float>(h,h);
                    cv::filter2D(h,h, CV_32F, kernel);

                    const int    hog_size = h.rows * h.cols;
                    const float *hog_ptr = h.ptr<float>();
                    for(int i = 0 ; i < hog_size ; ++i, ++dst_pos) {
                        dst_ptr[dst_pos] = hog_ptr[i] / max_magnitude;
                    }
                }
            } else if(params.has(Parameters::MAGNITUDE)) {
                cv::Mat magnitude;
                double norm = 1.0;
                if(params.normalize_magnitude) {
                    Magnitude::compute(src_as_float, magnitude, norm);
                } else {
                    Magnitude::compute(src_as_float, magnitude);
                }

                resample<float>(magnitude, magnitude);
                cv::filter2D(magnitude, magnitude, CV_32F, kernel);

                const int    magnitude_size = magnitude.rows * magnitude.cols;
                const float *magnitude_ptr = magnitude.ptr<float>();
                for(int i = 0 ; i < magnitude_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = magnitude_ptr[i] / norm;
                }
            }
            if(params.has(Parameters::LBP)) {
                cv::Mat lbp;
                cslibs_vision::LBP::standard(src_as_float, params.k, lbp);
                resample<uchar>(lbp,lbp);
                cv::filter2D(lbp, lbp, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    lbp_size = lbp.rows * lbp.cols;
                const float *lbp_ptr = lbp.ptr<float>();
                for(int i = 0 ; i < lbp_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = lbp_ptr[i] / norm;
                }
            }
            if(params.has(Parameters::LTP)) {
                cv::Mat ltp;
                cslibs_vision::LTP::standard(src_as_float, params.k, ltp);
                resample<uchar>(ltp, ltp);
                cv::filter2D(ltp, ltp, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    ltp_channels = ltp.channels();
                const float *ltp_ptr = ltp.ptr<float>();
                const int    ltp_step = ltp_channels * ltp.cols;
                for(int c = 0 ; c < ltp_channels ; ++c) {
                    for(int i = 0 ; i < ltp.rows ; ++i) {
                        for(int j = 0 ; j < ltp.cols ; ++j) {
                            dst_ptr[dst_pos] = ltp_ptr[i * ltp_step + j * ltp_channels + c] / norm;
                            ++dst_pos;
                        }
                    }
                }
            }
            if(params.has(Parameters::WLD)) {
                cv::Mat wld;
                cslibs_vision::WLD::standard(src_as_float, wld);

                resample<uchar>(wld, wld);
                cv::filter2D(wld, wld, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    wld_size = wld.cols * wld.rows;
                const float *wld_ptr = wld.ptr<float>();
                for(int i = 0 ; i  < wld_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = wld_ptr[i] / norm;
                }
            }
            if(params.has(Parameters::HOMOGENITY)) {
                cv::Mat homogenity;
                cslibs_vision::Homogenity::standard(homogenity, homogenity);

                resample<uchar>(homogenity, homogenity);
                cv::filter2D(homogenity, homogenity, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    homogenity_size = homogenity.cols * homogenity.rows;
                const float *homogenity_ptr = homogenity.ptr<float>();
                for(int i = 0 ; i  < homogenity_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = homogenity_ptr[i] / norm;
                }
            }
        } else {
            //// 3 channel
            cv::Mat gray_as_float(src_as_float.rows, src_as_float.cols, CV_32FC1, cv::Scalar());
            cv::cvtColor(src_as_float, gray_as_float, CV_BGR2GRAY);

            const std::size_t feature_size = params.featureSize(src.rows, src.cols);
            dst = cv::Mat(1, feature_size, CV_32FC1, cv::Scalar());
            float      *dst_ptr = dst.ptr<float>();
            std::size_t dst_pos = 0;

            if(params.has(Parameters::HOG)) {
                cv::Mat magnitude;
                double  max_magnitude;
                std::vector<cv::Mat> hog;
                HOG::standard(gray_as_float, params.hog_bin_size, hog, magnitude, max_magnitude);
                if(!params.normalize_magnitude) {
                    max_magnitude = 1.0;
                }

                if(params.has(Parameters::MAGNITUDE)) {
                    resample<float>(magnitude, magnitude);
                    cv::filter2D(magnitude, magnitude, CV_32F, kernel);

                    const int    magnitude_size = magnitude.rows * magnitude.cols;
                    const float *magnitude_ptr  = magnitude.ptr<float>();
                    for(int i = 0 ; i < magnitude_size ; ++i, ++dst_pos) {
                        dst_ptr[dst_pos] = magnitude_ptr[i] / max_magnitude;
                    }
                }
                for(cv::Mat &h : hog) {
                    resample<float>(h,h);
                    cv::filter2D(h,h, CV_32F, kernel);

                    const int    hog_size = h.rows * h.cols;
                    const float *hog_ptr = h.ptr<float>();
                    for(int i = 0 ; i < hog_size ; ++i, ++dst_pos) {
                        dst_ptr[dst_pos] = hog_ptr[i] / max_magnitude;
                    }
                }
            } else if(params.has(Parameters::MAGNITUDE)) {
                cv::Mat magnitude;
                double norm = 1.0;
                if(params.normalize_magnitude) {
                    Magnitude::compute(gray_as_float, magnitude, norm);
                } else {
                    Magnitude::compute(gray_as_float, magnitude);
                }

                resample<float>(magnitude, magnitude);
                cv::filter2D(magnitude, magnitude, CV_32F, kernel);

                const int    magnitude_size = magnitude.rows * magnitude.cols;
                const float *magnitude_ptr = magnitude.ptr<float>();
                for(int i = 0 ; i < magnitude_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = magnitude_ptr[i] / norm;
                }
            }
            if(params.has(Parameters::LUV)) {
                cv::Mat luv = cv::Mat(src.rows, src.cols, CV_32FC3, cv::Scalar());
                cv::cvtColor(src_as_float, luv, CV_BGR2Luv);
                resample<float>(luv, luv);
                cv::filter2D(luv, luv, CV_32F, kernel);

                const int luv_channels = luv.channels();
                const float *luv_ptr = luv.ptr<float>();
                const int luv_step = luv_channels * luv.cols;
                for(int c = 0 ; c < luv_channels ; ++c) {
                    for(int i = 0 ; i < luv.rows ; ++i) {
                        for(int j = 0 ; j < luv.cols ; ++j) {
                            dst_ptr[dst_pos] = luv_ptr[i * luv_step + j * luv_channels + c];
                            ++dst_pos;
                        }
                    }
                }
            }
            if(params.has(Parameters::LBP)) {
                cv::Mat lbp;
                cslibs_vision::LBP::standard(gray_as_float, params.k, lbp);
                resample<uchar>(lbp,lbp);
                cv::filter2D(lbp, lbp, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    lbp_size = lbp.rows * lbp.cols;
                const float *lbp_ptr = lbp.ptr<float>();
                for(int i = 0 ; i < lbp_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = lbp_ptr[i] / norm;
                }
            }
            if(params.has(Parameters::LTP)) {
                cv::Mat ltp;
                cslibs_vision::LTP::standard(gray_as_float, params.k, ltp);
                resample<uchar>(ltp, ltp);
                cv::filter2D(ltp, ltp, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    ltp_channels = ltp.channels();
                const float *ltp_ptr = ltp.ptr<float>();
                const int    ltp_step = ltp_channels * ltp.cols;
                for(int c = 0 ; c < ltp_channels ; ++c) {
                    for(int i = 0 ; i < ltp.rows ; ++i) {
                        for(int j = 0 ; j < ltp.cols ; ++j) {
                            dst_ptr[dst_pos] = ltp_ptr[i * ltp_step + j * ltp_channels + c] / norm;
                            ++dst_pos;
                        }
                    }
                }
            }
            if(params.has(Parameters::WLD)) {
                cv::Mat wld;
                cslibs_vision::WLD::standard(gray_as_float, wld);

                resample<uchar>(wld, wld);
                cv::filter2D(wld, wld, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    wld_size = wld.cols * wld.rows;
                const float *wld_ptr = wld.ptr<float>();
                for(int i = 0 ; i  < wld_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = wld_ptr[i] / norm;
                }
            }
            if(params.has(Parameters::HOMOGENITY)) {
                cv::Mat homogenity;
                cslibs_vision::Homogenity::standard(homogenity, homogenity);

                resample<uchar>(homogenity, homogenity);
                cv::filter2D(homogenity, homogenity, CV_32F, kernel);

                float norm = 1.f;
                if(params.normalize_patterns)
                    norm = 255.f;

                const int    homogenity_size = homogenity.cols * homogenity.rows;
                const float *homogenity_ptr = homogenity.ptr<float>();
                for(int i = 0 ; i  < homogenity_size ; ++i, ++dst_pos) {
                    dst_ptr[dst_pos] = homogenity_ptr[i] / norm;
                }
            }



        }
    }
};

}

#endif // ACF_HPP
