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

    /**
     * @brief The standard extraction mechanism.
     * @param _src      the image to extract the information of
     * @param k         the distance offset parameter
     */
    template <typename _Tp>
    static inline void stdExtraction(const cv::Mat &src,
                                     const _Tp k,
                                     cv::Mat &dst)
    {
        dst = cv::Mat_<int>(1, 256, 0);
        // calculate patterns
        for(int i=1;i<src.rows-1;++i) {
            for(int j=1;j<src.cols-1;++j) {
                _Tp center = src.at<_Tp>(i,j) + k;
                unsigned char histgram_pos = 0;

                histgram_pos += (src.at<_Tp>(i-1,j-1)   >= center) << 7;
                histgram_pos += (src.at<_Tp>(i-1,j)     >= center) << 6;
                histgram_pos += (src.at<_Tp>(i-1,j+1)   >= center) << 5;
                histgram_pos += (src.at<_Tp>(i,j+1)     >= center) << 4;
                histgram_pos += (src.at<_Tp>(i+1,j+1)   >= center) << 3;
                histgram_pos += (src.at<_Tp>(i+1,j)     >= center) << 2;
                histgram_pos += (src.at<_Tp>(i+1,j-1)   >= center) << 1;
                histgram_pos += (src.at<_Tp>(i,j-1)     >= center) << 0;

                dst.at<int>(histgram_pos)++;
            }
        }
    }

    /**
     * @brief Extended extraction using a radius for the neighbourhood.
     * @param _src          the source image
     * @param radius        the neighbourhood radius
     * @param neighbours    the neighbour count
     * @param k             the distance offset
     */
    template <typename _Tp>
    static inline void extExtraction(const cv::Mat &src,
                                     const int radius,
                                     const int neighbours,
                                     const _Tp k,
                                     cv::Mat &dst)
    {
        dst = cv::Mat_<int>(1, 256, 0);

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
                    _Tp center = src.at<_Tp>(i,j) + k;
                    unsigned char histogram_pos = 0;
                    histogram_pos += (t >= center) << n;
                    dst.at<int>(histogram_pos)++;
                }
            }
        }
    }


    template <typename _Tp>
    static inline void stdExtraction(const cv::Mat &src,
                                     const _Tp k,
                                     std::vector<int> &dst)
    {
        cv::Mat tmp;
        stdExtraction<_Tp>(src, k, tmp);
        tmp.copyTo(dst);
    }

    template <typename _Tp>
    static inline void extExtraction(const cv::Mat &src,
                                     const int radius,
                                     const int neighbours,
                                     const _Tp k,
                                     std::vector<int> &dst)
    {
        cv::Mat tmp;
        extExtraction<_Tp>(src, radius, neighbours, k, tmp);
        tmp.copyTo(dst);
    }
};
}
#endif // LBP_HPP

