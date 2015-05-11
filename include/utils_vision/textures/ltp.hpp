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
     * @brief LTP constructor.
     */
    LTP() :
        pos_(cv::Mat_<int>(1, 256, 0)),
        neg_(cv::Mat_<int>(1, 256, 0))
    {
    }

    /**
     * @brief operator - calculates the euclidian distance
     *        between two descriptors.
     * @param other     LTP histogram
     * @return          the distance
     */
    inline double operator - (const LTP &other) const
    {
        double sum = 0;
        for(int i = 0 ; i < pos_.rows ; ++i) {
            int pos_diff = pos_.at<int>(i) - other.pos_.at<int>(i);
            int neg_diff = neg_.at<int>(i) - other.neg_.at<int>(i);
            sum += pos_diff * pos_diff + neg_diff * neg_diff;
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Get the positive match histogram.
     * @param pos       a matrix to be written to
     */
    void getPos(cv::Mat &pos) const
    {
        pos_.copyTo(pos);
    }

    /**
     * @brief Get the positive match histogram.
     * @param pos       a vector to be written to
     */
    void getPos(std::vector<int> &pos) const
    {
        pos_.copyTo(pos);
    }


    /**
     * @brief Get the negative match histogram.
     * @param neg       a matrix to be written to
     */
    void getNeg(cv::Mat &neg) const
    {
        neg_.copyTo(neg);
    }

    /**
     * @brief Get the negative match histogram.
     * @param neg       a vector to be written to
     */
    void getNeg(std::vector<int> &neg) const
    {
        neg_.copyTo(neg);
    }

    /**
     * @brief Get the the histograms all together
     * @param all       a matrix to be written to
     */
    void getAll(cv::Mat &all) const
    {
        all = cv::Mat(1, pos_.cols + neg_.cols, CV_32SC1, cv::Scalar::all(0));
        cv::Mat pos(all.colRange(0, pos_.cols - 1));
        cv::Mat neg(all.colRange(pos_.cols, pos_.cols + neg_.cols - 1));
        getPos(pos);
        getNeg(neg);
    }

    /**
     * @brief Get the the histograms all together
     * @param all       a vector to be written to
     */
    void getAll(std::vector<int> &all) const
    {
        all.clear();
        all.insert(all.end(), pos_.begin<int>(), pos_.end<int>());
        all.insert(all.end(), neg_.begin<int>(), neg_.end<int>());
    }



    /**
     * @brief The standard extraction mechanism.
     * @param _src  the image to extract the information of
     * @param k     the distance offset parameter
     */
    template <typename _Tp>
    inline void stdExtraction(cv::InputArray _src, const _Tp k) {
        pos_.setTo(0);
        neg_.setTo(0);
        // get matrices
        cv::Mat src = _src.getMat();
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

                neg_.at<int>(hist_neg_it)++;
                pos_.at<int>(hist_pos_it)++;
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
    inline void extExtraction(cv::InputArray _src, const int radius, const int neighbours, const _Tp k) {
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
                    _Tp center = src.at<_Tp>(i,j);
                    unsigned char hist_neg_it = 0;
                    unsigned char hist_pos_it = 0;
                    _Tp center_minu_k = center - k;
                    _Tp center_plus_k = center + k;
                    hist_pos_it += (t   >= center_plus_k) << n;
                    hist_neg_it += (t    < center_minu_k) << n;
                    neg_.at<int>(hist_neg_it)++;
                    pos_.at<int>(hist_pos_it)++;
                }
            }
        }
    }

private:
    cv::Mat pos_;
    cv::Mat neg_;

};
}

#endif // LTP_HPP

