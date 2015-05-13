#ifndef LTP_HPP
#define LTP_HPP

#include "texture_descriptor.hpp"

namespace utils_vision {
/**
 * @brief The LTP class is used to calculate local ternary
 *        patterns;
 */
class LTP : public TextureDescriptor {
public:
    typedef cv::Ptr<TextureDescriptor> Ptr;

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

    static inline void standardCompact(const cv::Mat &src,
                                       const double k,
                                       cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _standardCompact<uchar>(src, k, dst);  break;
        case CV_8SC1: _standardCompact<char>(src, k, dst);   break;
        case CV_16UC1:_standardCompact<ushort>(src, k, dst); break;
        case CV_16SC1:_standardCompact<short>(src, k, dst);  break;
        case CV_32SC1:_standardCompact<int>(src, k, dst);    break;
        case CV_32FC1:_standardCompact<float>(src, k, dst);  break;
        case CV_64FC1:_standardCompact<double>(src, k, dst); break;
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
        dst = cv::Mat_<cv::Vec2b>(src.rows-2, src.cols-2, 0);

        for(int i = 1 ; i < src.rows-1 ;++i) {
            for(int j=1 ; j < src.cols-1; ++j) {
                cv::Vec2b &entry = dst.at<cv::Vec2b>(i,j);
                double center = src.at<_Tp>(i,j);
                unsigned char code_neg = 0;
                unsigned char code_pos = 0;
                double center_minu_k = center - k;
                double center_plus_k = center + k;

                code_pos |= (src.at<_Tp>(i-1,j-1)   >= center_plus_k) << 7;
                code_neg |= (src.at<_Tp>(i-1,j-1)    < center_minu_k) << 7;
                code_pos |= (src.at<_Tp>(i-1,j)     >= center_plus_k) << 6;
                code_neg |= (src.at<_Tp>(i-1,j)      < center_minu_k) << 6;
                code_pos |= (src.at<_Tp>(i-1,j+1)   >= center_plus_k) << 5;
                code_neg |= (src.at<_Tp>(i-1,j+1)    < center_minu_k) << 5;
                code_pos |= (src.at<_Tp>(i,j+1)     >= center_plus_k) << 4;
                code_neg |= (src.at<_Tp>(i,j+1)      < center_minu_k) << 4;
                code_pos |= (src.at<_Tp>(i+1,j+1)   >= center_plus_k) << 3;
                code_neg |= (src.at<_Tp>(i+1,j+1)    < center_minu_k) << 3;
                code_pos |= (src.at<_Tp>(i+1,j)     >= center_plus_k) << 2;
                code_neg |= (src.at<_Tp>(i+1,j)      < center_minu_k) << 2;
                code_pos |= (src.at<_Tp>(i+1,j-1)   >= center_plus_k) << 1;
                code_neg |= (src.at<_Tp>(i+1,j-1)    < center_minu_k) << 1;
                code_pos |= (src.at<_Tp>(i,j-1)     >= center_plus_k) << 0;
                code_neg |= (src.at<_Tp>(i,j-1)      < center_minu_k) << 0;
                entry[0] = code_neg;
                entry[1] = code_pos;
            }
        }
    }

    template <typename _Tp>
    static inline void _standardCompact(const cv::Mat &src,
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

        double thresholdneg = -k;
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);
        double diff = 0.0;
        double max  = 0.0;
        double min  = 255.0;

        for(int i=1 ; i < src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                _Tp center = src.at<_Tp>(i,j);
                unsigned char code = 0;

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
                dst.at<unsigned char>(i-1,j-1) = code;
                //cout<<(int)code<<" ";
                if(code > max)
                    max = (double) code;
                if(code<min)
                    min = (double) code;
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
        dst = cv::Mat_<cv::Vec2i>(src.rows-2*radius, src.cols-2*radius, 0);
        for(int i=0; i<n; ++i) {
            // sample points
            double x = static_cast<double>(radius) * cos(2.0*M_PI*i/static_cast<float>(n));
            double y = static_cast<double>(radius) * -sin(2.0*M_PI*i/static_cast<float>(n));
            // relative indices
            int fx = static_cast<int>(floor(x));
            int fy = static_cast<int>(floor(y));
            int cx = static_cast<int>(ceil(x));
            int cy = static_cast<int>(ceil(y));
            // fractional part
            float ty = y - fy;
            float tx = x - fx;
            // set interpolation weights
            double w1 = (1 - tx) * (1 - ty);
            double w2 =      tx  * (1 - ty);
            double w3 = (1 - tx) *      ty;
            double w4 =      tx  *      ty;
            // iterate through your data
            for(int i=radius; i < src.rows-radius;i++) {
                for(int j=radius;j < src.cols-radius;j++) {
                    double t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                    double cmk = src.at<_Tp>(i,j) - k;
                    double cpk = src.at<_Tp>(i,j) + k;
                    cv::Vec2i &entry = dst.at<cv::Vec2i>(i-radius, j-radius);
                    entry[0] += ((t >= cpk ) && (abs(t - cpk) > std::numeric_limits<double>::epsilon())) << i;
                    entry[1] += ((t <  cmk ) && (abs(t - cpk) > std::numeric_limits<double>::epsilon())) << i;
                }
            }
        }
    }
};
}

#endif // LTP_HPP

