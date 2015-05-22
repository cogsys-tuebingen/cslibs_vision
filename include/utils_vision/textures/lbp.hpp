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


    static inline void centerSymmetric(const cv::Mat &src,
                                       const double k,
                                       cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _centerSymmetric<uchar>(src, k, dst);  break;
        case CV_8SC1: _centerSymmetric<char>(src, k, dst);   break;
        case CV_16UC1:_centerSymmetric<ushort>(src, k, dst); break;
        case CV_16SC1:_centerSymmetric<short>(src, k, dst);  break;
        case CV_32SC1:_centerSymmetric<int>(src, k, dst);    break;
        case CV_32FC1:_centerSymmetric<float>(src, k, dst);  break;
        case CV_64FC1:_centerSymmetric<double>(src, k, dst); break;
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
        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar     *dst_ptr = dst.ptr<uchar>();

        int           prev, pos, next = 0;
        double        center = 0.0;
        unsigned char code = 0;

        for(int i = 1 ; i < src.rows-1 ;++i) {
            for(int j=1 ; j < src.cols-1; ++j) {
                prev = (i - 1) * src.cols + j;
                pos  = i * src.cols + j;
                next = (i + 1) * src.cols + j;
                center = src_ptr[pos] + k;
                code = 0;

                code |= (src_ptr[prev - 1] > center) << 7;
                code |= (src_ptr[prev]     > center) << 6;
                code |= (src_ptr[prev + 1] > center) << 5;

                code |= (src_ptr[pos  - 1] > center) << 0;
                code |= (src_ptr[pos  + 1] > center) << 4;

                code |= (src_ptr[next - 1] > center) << 1;
                code |= (src_ptr[next]     > center) << 2;
                code |= (src_ptr[next + 1] > center) << 3;

                dst_ptr[(i-1) * dst.cols + j - 1] = code;
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
        dst   = cv::Mat_<int>(src.rows-2*radius, src.cols-2*radius, (int) 0);

        const _Tp *src_ptr = src.ptr<_Tp>();
        int       *dst_ptr = dst.ptr<int>();

        double x, y, tx, ty, w1, w2, w3, w4, t= 0.0;
        int fx, fy, cx, cy, pos_cy, pos_fy, pos = 0;
        _Tp center = 0;

        for(int m=0; m<n; ++m) {
            // sample points
            x  = (double) radius * cos(2.0*M_PI*m / (double) n);
            y  = (double) radius *-sin(2.0*M_PI*m / (double) n);
            // relative indices
            fx = floor(x);
            fy = floor(y);
            cx = ceil(x);
            cy = ceil(y);
            // fractional part
            ty = y - fy;
            tx = x - fx;
            // set interpolation weights
            w1 = (1 - tx) * (1 - ty);
            w2 =      tx  * (1 - ty);
            w3 = (1 - tx) *      ty;
            w4 =      tx  *      ty;
            // iterate through your data
            for(int i=radius; i < src.rows-radius; ++i) {
                for(int j=radius;j < src.cols-radius; ++j) {
                    pos_cy = (i + cy) * src.cols + j;
                    pos_fy = (i + fy) * src.cols + j;
                    pos = (i * src.cols) + j;
                    t = w1 * src_ptr[pos_fy + fx] +
                            w2 * src_ptr[pos_fy + cx] +
                            w3 * src_ptr[pos_cy + fx] +
                            w4 * src_ptr[pos_cy + cx];

                    center = src_ptr[pos] + k;
                    // we are dealing with floating point precision, so add some little tolerance
                    dst_ptr[(i -radius) * dst.cols + j - radius]
                            += ((t > center) &&
                                (std::abs(t - (center) > std::numeric_limits<double>::epsilon()))) << m;
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

        int dst_rows = src.rows - 2*radius;
        int dst_cols = src.cols - 2*radius;
        dst    = cv::Mat_<float>(dst_rows, dst_cols, 0.f);
        cv::Mat _mean  = cv::Mat_<float>(src.rows, src.cols, 0.f);
        cv::Mat _delta = cv::Mat_<float>(src.rows, src.cols, 0.f);
        cv::Mat _m2    = cv::Mat_<float>(src.rows, src.cols, 0.f);

        const _Tp *src_ptr = src.ptr<_Tp>();
        float *_mean_ptr  = _mean.ptr<float>();
        float *_delta_ptr = _delta.ptr<float>();
        float *_m2_ptr    = _m2.ptr<float>();
        float *dst_ptr = dst.ptr<float>();

        double x, y, tx, ty, w1, w2, w3, w4, t= 0.0;
        int fx, fy, cx, cy, pos_cy, pos_fy, pos = 0;

        for(int m=0; m<n; ++m) {
            // sample points
            x  = (double) radius * cos(2.0*M_PI*m / (double) n);
            y  = (double) radius *-sin(2.0*M_PI*m / (double) n);
            // relative indices
            fx = floor(x);
            fy = floor(y);
            cx = ceil(x);
            cy = ceil(y);
            // fractional part
            ty = y - fy;
            tx = x - fx;
            // set interpolation weights
            w1 = (1 - tx) * (1 - ty);
            w2 =      tx  * (1 - ty);
            w3 = (1 - tx) *      ty;
            w4 =      tx  *      ty;
            // iterate through your data
            for(int i=radius; i < src.rows-radius;++i) {
                for(int j=radius;j < src.cols-radius;++j) {
                    pos_cy = (i + cy) * src.cols + j;
                    pos_fy = (i + fy) * src.cols + j;
                    pos = (i * src.cols) + j;
                    t = w1 * src_ptr[pos_fy + fx] +
                            w2 * src_ptr[pos_fy + cx] +
                            w3 * src_ptr[pos_cy + fx] +
                            w4 * src_ptr[pos_cy + cx];

                    _delta_ptr[pos] = t - _mean_ptr[pos];
                    _mean_ptr[pos] = _mean_ptr[pos] + (_delta_ptr[pos]) / (double) (i + 1);
                    _m2_ptr[pos] = _m2_ptr[pos] + _delta_ptr[pos] * (t - _mean_ptr[pos]);
                }
            }
        }
        // calculate result

        int i_dst = 0;
        int j_dst = 0;
        const float norm = n - 1;
        for(int i = radius; i < src.rows-radius; ++i,++i_dst) {
            j_dst = 0;
            for(int j = radius; j < src.cols-radius; ++j,++j_dst) {
                dst_ptr[i_dst * dst.cols + j_dst] = _m2_ptr[i * _m2.cols + j] / norm;
            }
        }
    }


    template <typename _Tp>
    static inline void _centerSymmetric(const cv::Mat& src,
                                        const double k,
                                        cv::Mat& dst)
    {
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);

        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar     *dst_ptr = dst.ptr<uchar>();
        double diff = 0.0;

        int           upper, lower = 0;
        unsigned char code = 0;

        for(int i = 1 ; i < src.rows-1 ;++i) {
            for(int j=1 ; j < src.cols-1; ++j) {
                upper = (i - 1) * src.cols + j-1;
                lower = (i + 1) * src.cols + j-1;

                code = 0;
                diff=src_ptr[upper]-src_ptr[lower + 2];
                code |= ( diff> k ) << 3;
                diff=src_ptr[upper+1]-src_ptr[lower+1];
                code |= ( diff> k ) << 2;
                diff=src_ptr[upper+2]-src_ptr[lower];
                code |= ( diff> k ) << 1;
                diff=src_ptr[upper + src.cols + 2]-src_ptr[upper+src.cols];
                code |= ( diff> k ) << 0;
                dst_ptr[(i-1) * dst.cols + j-1] = code;
            }
        }


    }



};
}
#endif // LBP_HPP

