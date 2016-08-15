#ifndef LTP_HPP
#define LTP_HPP

#include <opencv2/core/core.hpp>

namespace cslibs_vision {
/**
 * @brief The LTP class is used to calculate local ternary
 *        patterns;
 */
class LTP {
public:
    LTP() = delete;

    template <typename _Tp>
    static inline void histogram(const cv::Mat &src,
                                 const _Tp k,
                                 cv::Mat &dst) {
        dst = cv::Mat_<int>(1, 512, 0);
        cv::Mat pos = dst.colRange(0, 256);
        cv::Mat neg = dst.colRange(256, 512);

        // calculate patterns
        for(int i=1;i<src.rows-1;++i) {
            for(int j=1;j<src.cols-1;++j) {
                _Tp center = src.at<_Tp>(i,j);
                unsigned char hist_neg_it = 0;
                unsigned char hist_pos_it = 0;
                _Tp center_minu_k = center - k;
                _Tp center_plus_k = center + k;

                hist_pos_it |= (src.at<_Tp>(i-1,j-1)   >= center_plus_k) << 7;
                hist_neg_it |= (src.at<_Tp>(i-1,j-1)    < center_minu_k) << 7;
                hist_pos_it |= (src.at<_Tp>(i-1,j)     >= center_plus_k) << 6;
                hist_neg_it |= (src.at<_Tp>(i-1,j)      < center_minu_k) << 6;
                hist_pos_it |= (src.at<_Tp>(i-1,j+1)   >= center_plus_k) << 5;
                hist_neg_it |= (src.at<_Tp>(i-1,j+1)    < center_minu_k) << 5;
                hist_pos_it |= (src.at<_Tp>(i,j+1)     >= center_plus_k) << 4;
                hist_neg_it |= (src.at<_Tp>(i,j+1)      < center_minu_k) << 4;
                hist_pos_it |= (src.at<_Tp>(i+1,j+1)   >= center_plus_k) << 3;
                hist_neg_it |= (src.at<_Tp>(i+1,j+1)    < center_minu_k) << 3;
                hist_pos_it |= (src.at<_Tp>(i+1,j)     >= center_plus_k) << 2;
                hist_neg_it |= (src.at<_Tp>(i+1,j)      < center_minu_k) << 2;
                hist_pos_it |= (src.at<_Tp>(i+1,j-1)   >= center_plus_k) << 1;
                hist_neg_it |= (src.at<_Tp>(i+1,j-1)    < center_minu_k) << 1;
                hist_pos_it |= (src.at<_Tp>(i,j-1)     >= center_plus_k) << 0;
                hist_neg_it |= (src.at<_Tp>(i,j-1)      < center_minu_k) << 0;

                neg.at<int>(hist_neg_it)++;
                pos.at<int>(hist_pos_it)++;
            }
        }
    }

    template <typename _Tp>
    static inline void histogram(const cv::Mat &src,
                                 const _Tp k,
                                 std::vector<int> &dst) {
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


    static inline void shortened(const cv::Mat &src,
                                 const double k,
                                 cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _shortened<uchar>(src, k, dst);  break;
        case CV_8SC1: _shortened<char>(src, k, dst);   break;
        case CV_16UC1:_shortened<ushort>(src, k, dst); break;
        case CV_16SC1:_shortened<short>(src, k, dst);  break;
        case CV_32SC1:_shortened<int>(src, k, dst);    break;
        case CV_32FC1:_shortened<float>(src, k, dst);  break;
        case CV_64FC1:_shortened<double>(src, k, dst); break;
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

private:
    template <typename _Tp>
    inline static void _standard(const cv::Mat& src,
                                 const double k,
                                 cv::Mat& dst)
    {
        dst = cv::Mat_<cv::Vec2b>(src.rows-2, src.cols-2, cv::Vec2b());

        const _Tp *src_ptr = src.ptr<_Tp>();
        cv::Vec2b *dst_ptr = dst.ptr<cv::Vec2b>();

        int           prev, pos, next = 0;
        unsigned char code_neg = 0;
        unsigned char code_pos = 0;
        double center, center_minu_k, center_plus_k = 0.0;

        for(int i = 1 ; i < src.rows-1 ;++i) {
            for(int j=1 ; j < src.cols-1; ++j) {
                cv::Vec2b &entry = *dst_ptr;

                pos  = i * src.cols + j;
                prev = pos - src.cols;
                next = pos + src.cols;
                center = src_ptr[pos] + k;

                code_neg = 0;
                code_pos = 0;
                center_minu_k = center - k;
                center_plus_k = center + k;

                code_pos |= (src_ptr[prev - 1] >= center_plus_k) << 7;
                code_neg |= (src_ptr[prev - 1] < center_minu_k)  << 7;
                code_pos |= (src_ptr[prev]     >= center_plus_k) << 6;
                code_neg |= (src_ptr[prev]     < center_minu_k)  << 6;
                code_pos |= (src_ptr[prev + 1] >= center_plus_k) << 5;
                code_neg |= (src_ptr[prev + 1] < center_minu_k)  << 5;

                code_pos |= (src_ptr[pos  - 1] >= center_plus_k) << 0;
                code_neg |= (src_ptr[pos  - 1] < center_minu_k)  << 0;
                code_pos |= (src_ptr[pos  + 1] >= center_plus_k) << 4;
                code_neg |= (src_ptr[pos  + 1] < center_minu_k)  << 4;

                code_pos |= (src_ptr[next - 1] >= center_plus_k) << 1;
                code_neg |= (src_ptr[next - 1] < center_minu_k)  << 1;
                code_pos |= (src_ptr[next]     >= center_plus_k) << 2;
                code_neg |= (src_ptr[next]     < center_minu_k)  << 2;
                code_pos |= (src_ptr[next + 1] >= center_plus_k) << 3;
                code_neg |= (src_ptr[next + 1] < center_minu_k)  << 3;

                entry[0] = code_neg;
                entry[1] = code_pos;

                ++dst_ptr;
            }
        }
    }

    template <typename _Tp>
    static inline void _centerSymmetric(const cv::Mat& src,
                                        const double k,
                                        cv::Mat& dst)
    {
        dst = cv::Mat_<cv::Vec2b>(src.rows-2, src.cols-2, cv::Vec2b());

        const _Tp *src_ptr = src.ptr<_Tp>();
        cv::Vec2b *dst_ptr = dst.ptr<cv::Vec2b>();

        double diff = 0.0;
        int           upper, lower = 0;
        unsigned char code_neg = 0;
        unsigned char code_pos = 0;

        for(int i = 1 ; i < src.rows-1 ;++i) {
            for(int j=1 ; j < src.cols-1; ++j) {
                cv::Vec2b &entry = *dst_ptr;

                upper = (i - 1) * src.cols + j-1;
                lower = (i + 1) * src.cols + j-1;

                code_neg = 0;
                code_pos = 0;

                diff=src_ptr[upper]-src_ptr[lower + 2];
                code_pos |= (diff >= k) << 3;
                code_neg |= (diff <  k) << 3;
                diff=src_ptr[upper+1]-src_ptr[lower+1];
                code_pos |= (diff >= k) << 2;
                code_neg |= (diff <  k) << 2;
                diff=src_ptr[upper+2]-src_ptr[lower];
                code_pos |= (diff >= k) << 1;
                code_neg |= (diff <  k) << 1;
                diff=src_ptr[upper + src.cols + 2]-src_ptr[upper+src.cols];
                code_pos |= (diff >= k) << 0;
                code_neg |= (diff <  k) << 0;

                entry[0] = code_neg;
                entry[1] = code_pos;
                ++dst_ptr;
            }
        }
    }

    template <typename _Tp>
    static inline void _shortened(const cv::Mat &src,
                                  const double k,
                                  cv::Mat &dst)
    {
        static const unsigned char lut[8][3] = {{0,8,16},
                                                {1,9,17},
                                                {2,10,18},
                                                {3,11,19},
                                                {4,12,20},
                                                {5,13,21},
                                                {6,14,22},
                                                {7,15,23}};

        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);

        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar *dst_ptr = dst.ptr<uchar>();

        double thresholdneg = -k;
        double diff = 0.0;

        _Tp center = 0;
        unsigned char code = 0;

        for(int i=1 ; i < src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {

                code = 0;
                center = src_ptr[i * src.cols + j];

                //// ------------------------------
                diff=src.at<_Tp>(i-1,j-1) - center;
                if(diff>k)
                    code=lut[0][0];
                else if(diff<thresholdneg)
                    code=lut[0][2];
                else
                    code=lut[0][1];
                //// ------------------------------
                diff=src.at<_Tp>(i-1,j) - center;
                if(diff>k)
                    code+=lut[1][0];
                else if(diff<thresholdneg)
                    code+=lut[1][2];
                else
                    code+=lut[1][1];
                //// ------------------------------
                diff=src.at<_Tp>(i-1,j+1) - center;
                if(diff>k)
                    code+=lut[2][0];
                else if(diff<thresholdneg)
                    code+=lut[2][2];
                else
                    code+=lut[2][1];
                diff=src.at<_Tp>(i,j+1) - center;
                //// ------------------------------
                if(diff>k)
                    code+=lut[3][0];
                else if(diff<thresholdneg)
                    code+=lut[3][2];
                else
                    code+=lut[3][1];
                //// ------------------------------
                diff=src.at<_Tp>(i+1,j+1) - center;
                if(diff>k)
                    code+=lut[4][0];
                else if(diff<thresholdneg)
                    code+=lut[4][2];
                else
                    code+=lut[4][1];
                //// ------------------------------
                diff=src.at<_Tp>(i+1,j) - center;
                if(diff>k)
                    code+=lut[5][0];
                else if(diff<thresholdneg)
                    code+=lut[5][2];
                else
                    code+=lut[5][1];
                //// ------------------------------
                diff=src.at<_Tp>(i+1,j-1) - center;
                if(diff>k)
                    code+=lut[6][0];
                else if(diff<thresholdneg)
                    code+=lut[6][2];
                else
                    code+=lut[6][1];
                //// ------------------------------
                diff=src.at<_Tp>(i,j-1) - center;
                if(diff>k)
                    code+=lut[7][0];
                else if(diff<thresholdneg)
                    code+=lut[7][2];
                else
                    code+=lut[7][1];

                //// ------------------------------
                *dst_ptr = code;
                ++dst_ptr;

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
        dst = cv::Mat_<cv::Vec2i>(src.rows-2*radius, src.cols-2*radius, cv::Vec2i());

        const _Tp *src_ptr = src.ptr<_Tp>();
        cv::Vec2i *dst_ptr = dst.ptr<cv::Vec2i>();

        double x, y, tx, ty, w1, w2, w3, w4, cpk, cmk, t= 0.0;
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
            dst_ptr = dst.ptr<cv::Vec2i>();
            for(int i=radius; i < src.rows-radius;i++) {
                for(int j=radius;j < src.cols-radius;j++) {
                    cv::Vec2i &entry =  *dst_ptr;
                    pos_cy = (i + cy) * src.cols + j;
                    pos_fy = (i + fy) * src.cols + j;
                    pos = (i * src.cols) + j;
                    t = w1 * src_ptr[pos_fy + fx] +
                            w2 * src_ptr[pos_fy + cx] +
                            w3 * src_ptr[pos_cy + fx] +
                            w4 * src_ptr[pos_cy + cx];


                    cmk = src_ptr[pos] - k;
                    cpk = src_ptr[pos] + k;

                    entry[0] += ((t >= cpk ) && (abs(t - cpk) > std::numeric_limits<double>::epsilon())) << m;
                    entry[1] += ((t <  cmk ) && (abs(t - cpk) > std::numeric_limits<double>::epsilon())) << m;
                    ++dst_ptr;
                }
            }
        }
    }
};
}

#endif // LTP_HPP

