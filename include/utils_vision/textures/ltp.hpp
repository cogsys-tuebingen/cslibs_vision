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

    /**
     * @brief The standard extraction mechanism.
     * @param _src  the image to extract the information of
     * @param k     the distance offset parameter
     */
    template <typename _Tp>
    static inline void stdExtraction(const cv::Mat &src,
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

                hist_pos_it += (src.at<_Tp>(i-1,j-1)   >= center_plus_k) << 7;
                hist_neg_it += (src.at<_Tp>(i-1,j-1)    < center_minu_k) << 7;
                hist_pos_it += (src.at<_Tp>(i-1,j)     >= center_plus_k) << 6;
                hist_neg_it += (src.at<_Tp>(i-1,j)      < center_minu_k) << 6;
                hist_pos_it += (src.at<_Tp>(i-1,j+1)   >= center_plus_k) << 5;
                hist_neg_it += (src.at<_Tp>(i-1,j+1)    < center_minu_k) << 5;
                hist_pos_it += (src.at<_Tp>(i,j+1)     >= center_plus_k) << 4;
                hist_neg_it += (src.at<_Tp>(i,j+1)      < center_minu_k) << 4;
                hist_pos_it += (src.at<_Tp>(i+1,j+1)   >= center_plus_k) << 3;
                hist_neg_it += (src.at<_Tp>(i+1,j+1)    < center_minu_k) << 3;
                hist_pos_it += (src.at<_Tp>(i+1,j)     >= center_plus_k) << 2;
                hist_neg_it += (src.at<_Tp>(i+1,j)      < center_minu_k) << 2;
                hist_pos_it += (src.at<_Tp>(i+1,j-1)   >= center_plus_k) << 1;
                hist_neg_it += (src.at<_Tp>(i+1,j-1)    < center_minu_k) << 1;
                hist_pos_it += (src.at<_Tp>(i,j-1)     >= center_plus_k) << 0;
                hist_neg_it += (src.at<_Tp>(i,j-1)      < center_minu_k) << 0;

                neg.at<int>(hist_neg_it)++;
                pos.at<int>(hist_pos_it)++;
            }
        }
    }

    /**
     * @brief Extended extraction using a radius for the neighbourhood.
     * @param _src          the source image
     * @param radius        the neighbourhood adius
     * @param neighbors     the neighour count
     * @param k             the distance offset
     */
    template <typename _Tp>
    static inline void extExtraction(const cv::Mat  &src,
                                     const int radius,
                                     const int neighbours,
                                     const _Tp k,
                                     cv::Mat &dst) {

        dst = cv::Mat_<int>(1, 512, 0);
        cv::Mat pos = dst.colRange(0, 256);
        cv::Mat neg = dst.colRange(256, 512);

        for(int n=0; n<neighbours; ++n) {
            // sample points
            float x = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbours)));
            float y = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbours)));
            // relative indices
            int fx = static_cast<int>(floor(x));
            int fy = static_cast<int>(floor(y));
            int cx = static_cast<int>(ceil(x));
            int cy = static_cast<int>(ceil(y));
            // fractional part
            float ty = y - fy;
            float tx = x - fx;
            // set interpolation weights
            float w1 = (1 - tx) * (1 - ty);
            float w2 =      tx  * (1 - ty);
            float w3 = (1 - tx) *      ty;
            float w4 =      tx  *      ty;
            // iterate through your data
            for(int i=radius; i < src.rows-radius;++i) {
                for(int j=radius;j < src.cols-radius;++j) {
                    // calculate interpolated value
                    float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                    // floating point precision, so check some machine-dependent epsilon
                    _Tp center = src.at<_Tp>(i,j);
                    unsigned char hist_neg_it = 0;
                    unsigned char hist_pos_it = 0;
                    _Tp center_minu_k = center - k;
                    _Tp center_plus_k = center + k;
                    hist_pos_it += (t   >= center_plus_k) << n;
                    hist_neg_it += (t    < center_minu_k) << n;
                    neg.at<int>(hist_neg_it)++;
                    pos.at<int>(hist_pos_it)++;
                }
            }
        }
    }

    template <typename _Tp>
    static inline void stdExtraction(const cv::Mat &src,
                                     const _Tp k,
                                     std::vector<int> &dst) {
        cv::Mat tmp;
        stdExtraction<_Tp>(src, k, tmp);
        tmp.copyTo(dst);
    }

    template <typename _Tp>
    static inline void extExtraction(const cv::Mat  &src,
                                     const int radius,
                                     const int neighbours,
                                     const _Tp k,
                                     std::vector<int> &dst) {
        cv::Mat tmp;
        extExtraction<_Tp>(src, radius, neighbours, k, tmp);
        tmp.copyTo(dst);
    }

};
}

#endif // LTP_HPP

