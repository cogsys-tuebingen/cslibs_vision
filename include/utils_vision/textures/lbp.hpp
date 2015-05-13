#ifndef LBP_HPP
#define LBP_HPP

#include "texture_descriptor.hpp"

namespace utils_vision {
/**
 * @brief The LTP class is used to calculate local binary
 *        patterns;
 */
class LBP : public TextureDescriptor
{
public:
    typedef cv::Ptr<LBP> Ptr;

    template <typename _Tp>
    static inline void histogram(const cv::Mat &src,
                                 const double k,
                                 cv::Mat &dst)
    {
        dst = cv::Mat_<int>(1, 256, 0);
        // calculate patterns
        for(int i=1;i<src.rows-1;++i) {
            for(int j=1;j<src.cols-1;++j) {
                double center = src.at<_Tp>(i,j) + k;
                unsigned char histgram_pos = 0;

                histgram_pos |= (src.at<_Tp>(i-1,j-1)   >= center) << 7;
                histgram_pos |= (src.at<_Tp>(i-1,j)     >= center) << 6;
                histgram_pos |= (src.at<_Tp>(i-1,j+1)   >= center) << 5;
                histgram_pos |= (src.at<_Tp>(i,j+1)     >= center) << 4;
                histgram_pos |= (src.at<_Tp>(i+1,j+1)   >= center) << 3;
                histgram_pos |= (src.at<_Tp>(i+1,j)     >= center) << 2;
                histgram_pos |= (src.at<_Tp>(i+1,j-1)   >= center) << 1;
                histgram_pos |= (src.at<_Tp>(i,j-1)     >= center) << 0;

                dst.at<int>(histgram_pos)++;
            }
        }
    }

    template <typename _Tp>
    static inline void histogram(const cv::Mat &src,
                            const _Tp k,
                            std::vector<int> &dst)
    {
        cv::Mat tmp;
        histogram<_Tp>(src, k, tmp);
        tmp.copyTo(dst);
    }

    static inline void standard(const cv::Mat &src,
                                const double k,
                                cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _standard<uchar>(src, k, dst);  break;
        case CV_8SC1: _standard<char>(src, k, dst);   break;
        case CV_16UC1:_standard<ushort>(src, k, dst); break;
        case CV_16SC1:_standard<short>(src, k, dst);  break;
        case CV_32SC1:_standard<int>(src, k, dst);    break;
        case CV_32FC1:_standard<float>(src, k, dst);  break;
        case CV_64FC1:_standard<double>(src, k, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

    static inline void extended(const cv::Mat &src,
                                const int radius,
                                const int neighbours,
                                const double k,
                                cv::Mat &dst)
    {

        switch(src.type()) {
        case CV_8UC1: _extended<uchar>(src, radius, neighbours, k, dst);  break;
        case CV_8SC1: _extended<char>(src, radius, neighbours, k, dst);   break;
        case CV_16UC1:_extended<ushort>(src, radius, neighbours, k, dst); break;
        case CV_16SC1:_extended<short>(src, radius, neighbours, k, dst);  break;
        case CV_32SC1:_extended<int>(src, radius, neighbours, k, dst);    break;
        case CV_32FC1:_extended<float>(src, radius, neighbours, k, dst);  break;
        case CV_64FC1:_extended<double>(src, radius, neighbours, k, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

    static inline void var(const cv::Mat &src,
                           const int radius,
                           const int neighbours,
                           cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _var<uchar>(src, radius, neighbours, dst);  break;
        case CV_8SC1: _var<char>(src, radius, neighbours, dst);   break;
        case CV_16UC1:_var<ushort>(src, radius, neighbours, dst); break;
        case CV_16SC1:_var<short>(src, radius, neighbours, dst);  break;
        case CV_32SC1:_var<int>(src, radius, neighbours, dst);    break;
        case CV_32FC1:_var<float>(src, radius, neighbours, dst);  break;
        case CV_64FC1:_var<double>(src, radius, neighbours, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }


    static inline void center_symmetric(const cv::Mat &src,
                                        const double k,
                                        cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _center_symmetric<uchar>(src, k, dst);  break;
        case CV_8SC1: _center_symmetric<char>(src, k, dst);   break;
        case CV_16UC1:_center_symmetric<ushort>(src, k, dst); break;
        case CV_16SC1:_center_symmetric<short>(src, k, dst);  break;
        case CV_32SC1:_center_symmetric<int>(src, k, dst);    break;
        case CV_32FC1:_center_symmetric<float>(src, k, dst);  break;
        case CV_64FC1:_center_symmetric<double>(src, k, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

private:
    template <typename _Tp>
    inline static void _standard(const cv::Mat& src,
                                 const double k,
                                 cv::Mat& dst)
    {
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);

        for(int i = 1 ; i < src.rows-1 ;++i) {
            for(int j=1 ; j < src.cols-1; ++j) {
                double center = src.at<_Tp>(i,j) + k;
                unsigned char code = 0;
                code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
                code |= (src.at<_Tp>(i-1,j) > center)   << 6;
                code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
                code |= (src.at<_Tp>(i,j+1) > center)   << 4;
                code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
                code |= (src.at<_Tp>(i+1,j) > center)   << 2;
                code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
                code |= (src.at<_Tp>(i,j-1) > center)   << 0;
                dst.at<unsigned char>(i-1,j-1) = code;
            }
        }
    }


    template <typename _Tp>
    static inline void _extended(const cv::Mat& src,
                                 const int radius,
                                 const int neighbours,
                                 const double k,
                                 cv::Mat& dst) {
        int n = std::max(std::min(neighbours,31),1); // set bounds...
        dst = cv::Mat_<int>(src.rows-2*radius, src.cols-2*radius, 0);
        for(int i=0; i<n; ++i) {
            // sample points
            double x = static_cast<float>(radius) * cos(2.0*M_PI*i/static_cast<float>(n));
            double y = static_cast<float>(radius) * -sin(2.0*M_PI*i/static_cast<float>(n));
            // relative indices
            int fx = static_cast<int>(floor(x));
            int fy = static_cast<int>(floor(y));
            int cx = static_cast<int>(ceil(x));
            int cy = static_cast<int>(ceil(y));
            // fractional part
            double ty = y - fy;
            double tx = x - fx;
            // set interpolation weights
            double w1 = (1 - tx) * (1 - ty);
            double w2 =      tx  * (1 - ty);
            double w3 = (1 - tx) *      ty;
            double w4 =      tx  *      ty;
            // iterate through your data
            for(int i=radius; i < src.rows-radius;i++) {
                for(int j=radius;j < src.cols-radius;j++) {
                    float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                    // we are dealing with floating point precision, so add some little tolerance
                    dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j) + k) && (abs(t-(src.at<_Tp>(i,j) + k)) > std::numeric_limits<float>::epsilon())) << i;
                }
            }
        }
    }

    template <typename _Tp>
    static inline void _var(const cv::Mat& src,
                            const int radius,
                            const int neighbours,
                            cv::Mat& dst) {
        int n = std::max(std::min(neighbours,31),1); // set bounds
                dst    = cv::Mat_<float>(src.rows-2*radius, src.cols-2*radius, 0.f);
        cv::Mat _mean  = cv::Mat_<float>(src.rows, src.cols, 0.f);
        cv::Mat _delta = cv::Mat_<float>(src.rows, src.cols, 0.f);
        cv::Mat _m2    = cv::Mat_<float>(src.rows, src.cols, 0.f);
        for(int i=0; i<n; i++) {
            // sample points
            double x = static_cast<float>(radius) * cos(2.0*M_PI*i/static_cast<float>(n));
            double y = static_cast<float>(radius) * -sin(2.0*M_PI*i/static_cast<float>(n));
            // relative indices
            int fx = static_cast<int>(floor(x));
            int fy = static_cast<int>(floor(y));
            int cx = static_cast<int>(ceil(x));
            int cy = static_cast<int>(ceil(y));
            // fractional part
            double ty = y - fy;
            double tx = x - fx;
            // set interpolation weights
            double w1 = (1 - tx) * (1 - ty);
            double w2 =      tx  * (1 - ty);
            double w3 = (1 - tx) *      ty;
            double w4 =      tx  *      ty;
            // iterate through your data
            for(int i=radius; i < src.rows-radius;i++) {
                for(int j=radius;j < src.cols-radius;j++) {
                    double t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                    _delta.at<float>(i,j) = t - _mean.at<float>(i,j);
                    _mean.at<float>(i,j) = (_mean.at<float>(i,j) + (_delta.at<float>(i,j) / (1.0*(i+1)))); // i am a bit paranoid
                    _m2.at<float>(i,j) = _m2.at<float>(i,j) + _delta.at<float>(i,j) * (t - _mean.at<float>(i,j));
                }
            }
        }
        // calculate result
        for(int i = radius; i < src.rows-radius; i++) {
            for(int j = radius; j < src.cols-radius; j++) {
                dst.at<float>(i-radius, j-radius) = _m2.at<float>(i,j) / (1.0*(n-1));
            }
        }
    }


    template <typename _Tp>
    static inline void _center_symmetric(const cv::Mat& src,
                                         const double k,
                                         cv::Mat& dst)
    {
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);
        double diff = 0.0;
        for(int i=1 ; i<src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                //_Tp center = src.at<_Tp>(i,j);
                unsigned char code = 0;
                diff=src.at<_Tp>(i-1,j-1)-src.at<_Tp>(i+1,j+1);
                code |= ( diff> k ) << 3;
                diff=src.at<_Tp>(i-1,j)-src.at<_Tp>(i+1,j);
                code |= ( diff> k ) << 2;
                diff=src.at<_Tp>(i-1,j+1)-src.at<_Tp>(i+1,j-1);
                code |= ( diff> k ) << 1;
                diff=src.at<_Tp>(i,j+1)-src.at<_Tp>(i,j-1) ;
                code |= ( diff> k ) << 0;
                dst.at<unsigned char>(i-1,j-1) = code;
            }
        }
    }



};
}
#endif // LBP_HPP

