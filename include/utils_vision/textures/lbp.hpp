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
     * @brief LBP constructor.
     */
    LBP() :
        histogram_(cv::Mat_<int>(1, 256, 0))
    {
    }

    /**
     * @brief operator - calculates the euclidian distance
     *        between two descriptors.
     * @param other     LBP histogram
     * @return          the distance
     */
    double operator - (const LBP &other) const
    {
        double sum = 0;
        for(int i = 0 ; i < histogram_.rows ; ++i) {
            int diff = histogram_.at<int>(i) - other.histogram_.at<int>(i);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Get the calculated histogram.
     * @param histogram     a matrix to be written to
     */
    void get(cv::Mat &histogram) const
    {
        histogram_.copyTo(histogram);
    }

    /**
     * @brief Get the calculated histogram.
     * @param _histogram    a vector to be written to
     */
    void get(std::vector<int> &_histogram) const
    {
        histogram_.copyTo(_histogram);
    }

    /**
     * @brief The standard extraction mechanism.
     * @param _src      the image to extract the information of
     * @param k         the distance offset parameter
     */
    template <typename _Tp>
    inline void stdExtraction(cv::InputArray _src, const _Tp k = 0) {
        histogram_.setTo(0);
        // get matrices
        cv::Mat src = _src.getMat();
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

                histogram_.at<int>(histgram_pos)++;
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
    inline void extExtraction(cv::InputArray _src, const int radius, const int neighbours, const _Tp k = 0) {
        histogram_.setTo(0);
        //get matrices
        cv::Mat src = _src.getMat();
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
                    histogram_.at<int>(histogram_pos)++;
                }
            }
        }
    }


private:
    cv::Mat histogram_;
};
}
#endif // LBP_HPP

