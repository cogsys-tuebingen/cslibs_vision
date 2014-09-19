#ifndef CV_HISTOGRAM_HPP
#define CV_HISTOGRAM_HPP
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/assign.hpp>

/**
 * @namespace cv_histogram is a namespace containing functions that make calculating histograms easier.
 * @author Hanten, Richard
 */
namespace utils_vision {
namespace histogram {
const cv::Scalar COLOR_BLUE    = cv::Scalar(255, 0, 0);
const cv::Scalar COLOR_GREEN   = cv::Scalar(0, 255, 0);
const cv::Scalar COLOR_RED     = cv::Scalar(0, 0, 255);
const cv::Scalar COLOR_CYAN    = cv::Scalar(255, 255,0);
const cv::Scalar COLOR_WHITE   = cv::Scalar(255,255,255);
const cv::Scalar COLOR_YELLOW  = cv::Scalar(0,255,255);
const cv::Scalar COLOR_MAGENTA = cv::Scalar(255,0,255);
const std::vector<cv::Scalar>  COLOR_PALETTE = boost::assign::list_of
        (COLOR_BLUE)
        (COLOR_GREEN)
        (COLOR_RED)
        (COLOR_CYAN)
        (COLOR_WHITE)
        (COLOR_YELLOW)
        (COLOR_MAGENTA);

/**
 * @brief Normalize a rgb image.
 * @param src       the source image
 * @param dst       the destination image
 */
inline void normalize_rgb(const cv::Mat &src, cv::Mat &dst)
{
    cv::Vec3d mean;
    cv::Vec3d vari;

    dst = src.clone();

    cv::meanStdDev(src, mean, vari);
    cv::sqrt(vari, vari);

    uchar* dst_ptr = (uchar*) dst.data;

    for(int y = 0 ; y < src.rows ; y++) {
        for(int x = 0 ; x < src.cols ; x++) {
            for(int c = 0 ; c < src.channels() ; c++) {
                double val = dst_ptr[y *  src.step + x * src.channels() + c];
                val = (val - mean[c]) / vari[c] * 3 + 127;
                dst_ptr[y *  src.step + x * src.channels() + c] = val;
            }
        }
    }
}

/**
 * @brief Prepare parameters
 * @param bins
 * @param ranges
 * @param channel_count
 */
inline void prepare_params(cv::Mat &bins, cv::Mat &ranges, const int channel_count)
{
    bins = cv::Mat_<int>(channel_count, 1);
    ranges = cv::Mat_<float>(channel_count * 2, 1);

    for(int i = 0 ; i < channel_count ; i++) {
        bins.at<int>(i) = 256;
        ranges.at<float>(2 * i) = 0.f;
        ranges.at<float>(2 * i + 1) = 256.f;
    }
}

typedef std::pair<float,float> Range;

/**
 * @brief Return a Range struct containing the upper and lower boundary of a type
 *        defined interval.
 * @return          a Range struct
 */
template<typename Tp>
inline Range make_range()
{
    return std::make_pair<float, float>(std::numeric_limits<Tp>::min(), std::numeric_limits<Tp>::max());
}

/**
 * @brief Create a Range struct depending on the maximum and minimum value
 *        of a given data set.
 * @param src       the data set as a cv matrix
 * @param mask      a mask to rule out invalid set entries
 * @return          a Range struct
 */
template<typename Tp>
inline Range make_min_max_range(const cv::Mat &src,
                                const cv::Mat &mask = cv::noArray())
{
    double min_val = std::numeric_limits<Tp>::min();
    double max_val = std::numeric_limits<Tp>::max();
    if(src.channels() > 1) {
        cv::Mat tmp = src.clone();
        tmp.reshape(1);
        cv::minMaxLoc(tmp, &min_val, &max_val);
        if(!mask.empty())
            std::cerr << " No mask support for multi channel images!" << std::endl;
    } else {
        cv::minMaxLoc(src, &min_val, &max_val, NULL, NULL, mask);
    }
    return std::make_pair<float, float>(min_val, max_val);
}

/**
 * @brief Create several histograms.
 * @param channels      the matrices to be processed
 * @param histograms    the histograms to be returned
 * @param mask          a mask to rule out invalid data entries
 * @param bins          the amount of bins to be used
 * @param ranges        the ranges which should be observed
 * @param uniform       if the histogram should be uniform
 * @param accumulate    if accumulation should be used
 */
inline void histogram(const std::vector<cv::Mat>  &channels, std::vector<cv::Mat> &histograms, const cv::Mat &mask,
                      const std::vector<int> &bins, const std::vector<Range> &ranges,
                      bool uniform = true, bool accumulate = false)
{
    assert(bins.size() == channels.size());
    assert(ranges.size() == channels.size());

    uint channel_count = channels.size();
    histograms.resize(channel_count);

    for(uint i = 0 ; i < channel_count ; i++) {
        int   ch[] = {0};
        int   b[] = {bins.at(i)};
        std::vector<float*> r_ch;
        r_ch.push_back((float*) &(ranges.at(i)));
        cv::calcHist(&channels.at(i), 1, ch, mask, histograms.at(i), 1, b, (const float **) r_ch.data(), uniform , accumulate);
    }
}

/**
 * @brief Create a multi channel histogram represented by several histogram.s
 * @param src           the src matrix to be process
 * @param histograms    the vector the histograms will be stored in
 * @param mask          a mask to rule out invalid data entries
 * @param bins          the amount of bins to be used
 * @param ranges        the ranges which should be used
 * @param uniform       if the histograms should be uniform
 * @param accumulate    if accumulation should be used
 */
inline void histogram(const cv::Mat &src, std::vector<cv::Mat> &histograms, const cv::Mat &mask,
                      const std::vector<int> &bins, const std::vector<Range> &ranges,
                      bool uniform = true, bool accumulate = false)
{
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    histogram(channels, histograms, mask, bins, ranges, uniform, accumulate);
}

/**
 * @brief Do a histogram analysis for all channels of an image.
 * @param src           source image
 * @param histograms    the histograms will be written to here
 * @param mask          am mask for the analysis
 * @param bins          the amount of bins per channel
 * @param ranges        the ranges per channel
 * @param uniform       if the histogram should be uniform
 * @param accumulate    if the histogram computation shall work accumluative
 */
inline void histogram(const cv::Mat &src, std::vector<cv::MatND> &histograms, const cv::Mat &mask,
                      const cv::Mat &bins, const cv::Mat &ranges,
                      bool uniform = true, bool accumulate = false)
{
    assert(bins.rows == src.channels() || bins.cols == src.channels());
    assert(ranges.rows == src.channels() * 2 || ranges.cols == src.channels() * 2);

    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    uint channel_count = channels.size();
    histograms.resize(channel_count);

    for(uint i = 0 ; i < channel_count ; i++) {
        int   ch[] = {0};
        int   b[] = {bins.at<int>(i)};
        float r_ch_values[] = {ranges.at<float>(2 * i), ranges.at<float>(i * 2 + 1)};
        const float* r_ch[] = {r_ch_values};
        cv::calcHist(&channels[i], 1, ch, mask, histograms[i], 1, b, r_ch, uniform , accumulate);
    }
}

/**
 * @brief Do a full channel histogram equalization of an image.
 * @param src   source image
 * @param dst   destination image
 */
inline void full_channel_equalize(const cv::Mat &src, cv::Mat &dst)
{
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for(uint i = 0 ; i < channels.size() ; i++) {
        cv::equalizeHist(channels[i], channels[i]);
    }
    cv::merge(channels, dst);
}

/**
 * @brief Normalize all channesl of a matrix e.g. histogram matrix.
 * @param src               the source matrix
 * @param dst               the normalized matrix
 * @param channel_factors   list of <min, max> for limiting
 */
template<int norm>
inline void normalize(const cv::Mat &src, cv::Mat &dst, const std::vector<double> channel_factors)
{
    assert(channel_factors.size() % 2 == 0);

    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for(int i = 0 ; i < channel_factors.size() / 2 ; i += 2) {
        cv::normalize(channels[i], channels[i], channel_factors[i], channel_factors[i+1], norm);
    }
    cv::merge(channels, dst);
}

/**
 * @brief Retrieve maximum as preferred type.
 * @param src   the matrix to search the maximum in
 * @return
 */
template<typename T>
T maximum(const cv::Mat &src)
{
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    return max;
}

/**
 * @brief Render histograms as image to watch.
 * @param histograms        the histograms to render;
 * @param bins              the amount of bins
 * @param histogram_color   the histogram specific color to be used
 * @param dst               the image to write to
 */
template<class _Tp>
inline void render_histogram(const std::vector<cv::Mat> &histograms, const std::vector<int> bins,
                             const std::vector<cv::Scalar> histogram_colors, cv::Mat &dst, const double scale = 1.0)
{
    assert(!dst.empty());
    assert(bins.size() == histograms.size());
    assert(bins.size() < histogram_colors.size());

    double scale_;
    if(scale > 0) {
        scale_ = scale;
    }
    if(scale == 0) {
        scale_ = 0.1;
    }
    if(scale < 0) {
        scale_ = 1 / (double) std::abs(scale);
    }


    for(uint i = 0 ; i < histograms.size() ; i++) {
        cv::Mat histogram = histograms[i].clone();
        cv::normalize(histogram, histogram, dst.rows * scale_, cv::NORM_MINMAX);
        int bin_w = cvRound( dst.cols/ (double) bins[i] );
        for( int j = 1; j < bins[i]; j++ )
        {
            cv::line(dst,
                     cv::Point(bin_w*(j-1), dst.rows - cvRound(histogram.at<_Tp>(j - 1))),
                     cv::Point(bin_w*j, dst.rows - cvRound(histogram.at<_Tp>(j))),
                     histogram_colors[i]);
        }
    }
}

/**
 * @brief Render a curve representing a histogram.
 * @param src           the data set
 * @param color         the color to be used
 * @param line_width    the line width to be used
 * @param dst           the matrix the line should be rendered to
 */
template<typename Tp>
inline void render_curve(const cv::Mat &src, const cv::Scalar &color,
                         const int line_width,
                         cv::Mat &dst)
{
    cv::Mat tmp = src.clone();
    cv::normalize(tmp, tmp, dst.rows - 1, cv::NORM_MINMAX);
    int     bins        = std::max(tmp.rows, tmp.cols);
    double  bin_width   = dst.cols / (double) bins;
    double  offset      = bin_width / 2.0;

    for(int i = 1 ; i < bins ; ++i) {
        cv::line(dst,
                 cv::Point2f(bin_width * (i-1) + offset, dst.rows - tmp.at<Tp>(i - 1)),
                 cv::Point2f(bin_width * i + offset,     dst.rows - tmp.at<Tp>(i)),
                 color,
                 line_width,
                 CV_AA);
    }
}

/**
 * @brief Maximum typedef for pair of integer as the bin number and
 *        a float as the value of the bin.
 */
typedef std::pair<uint, float> Maximum;

/**
 * @brief Render a histogram as a curve with maxima position.
 * @param src           the histogram
 * @param maxima        the list of maxima found within
 * @param color         the color of the histogram graph
 * @param line_width    the line width to be used
 * @param radius        the radius of the maxima markers
 * @param dst           the image to render the curve to
 */
template<typename Tp>
inline void render_curve(const cv::Mat &src, const std::vector<Maximum> &maxima,
                         const cv::Scalar &color,
                         const int line_width, const int radius,
                         cv::Mat &dst)
{
    cv::Mat tmp = src.clone();
    cv::normalize(tmp, tmp, dst.rows - 1, cv::NORM_MINMAX);
    int     bins        = std::max(tmp.rows, tmp.cols);
    double  bin_width   = dst.cols / (double) bins;
    double  offset      = bin_width / 2.0;

    for(int i = 1 ; i < bins ; ++i) {
        cv::line(dst,
                 cv::Point2f(bin_width * (i-1) + offset, dst.rows - tmp.at<Tp>(i - 1)),
                 cv::Point2f(bin_width * i + offset,     dst.rows - tmp.at<Tp>(i)),
                 color,
                 line_width,
                 CV_AA);
    }

    for(std::vector<Maximum>::const_iterator it = maxima.begin() ;
        it != maxima.end() ;
        ++it) {
        cv::Point2f p(bin_width * it->first + offset,
                     dst.rows - tmp.at<Tp>(it->first));
        cv::circle(dst, p, radius, cv::Scalar(color[2], color[0], color[1]), line_width, CV_AA);
    }

}

/**
 * @brief Simple constraint check for a histogram entry being a local maximum.
 * @param src       the 1D function vector to be checked
 * @param i_min     the minimum index for the search
 * @param i         the point to be checked
 * @param i_max     the maximum index for the search
 * @param thresh    a threshold for minimum distance in the neighbourhood
 * @return
 */
template<typename T>
inline bool is_max(const cv::Mat &src,
                   const unsigned int i_min,
                   const unsigned int i,
                   const unsigned int i_max,
                   const T thresh)
{
    bool is_max = true;
    T value     = src.at<T>(i);
    T diff_left  = std::numeric_limits<T>::min();
    T diff_right = std::numeric_limits<T>::min();
    for(unsigned int it = i_min ; it < i ; ++it) {
        T prev = src.at<T>(it);
        is_max &= prev < value;
        diff_left = std::max(diff_left, std::abs(value - prev));
    }
    for(unsigned int it = i + 1 ; it <= i_max ; ++it) {
        T next = src.at<T>(it);
        is_max &= next < value;
        diff_right = std::max(diff_right, std::abs(next - value));
    }

    return is_max && (diff_right > thresh) && (diff_left > thresh);
}

template<typename T>
inline void find_maxima1D(const cv::Mat &src,
                          const unsigned int k,
                          const T thresh,
                          std::vector<Maximum> &dst)
{
    assert(src.cols == 1);
    if(k > src.rows)
        return;

    /// upper
    for(unsigned int i = 0 ; i < k ; ++i) {
        if(is_max<T>(src, 0, i, i + k, thresh)) {
            dst.push_back(std::make_pair<uint, float>(i, src.at<T>(i)));
        }
    }

    /// inner
    for(unsigned int i = k ; i < src.rows - k ; ++i) {
        if(is_max<T>(src, i - k, i , i + k, thresh)) {
            dst.push_back(std::make_pair<uint, float>(i, src.at<T>(i)));
        }
    }

    /// lower
    for(unsigned int i = src.rows - 1 - k ; i < src.rows ; ++i) {
        if(is_max<T>(src, i - k , i , src.rows - 1, thresh)) {
            dst.push_back(std::make_pair<uint, float>(i, src.at<T>(i)));
        }
    }
}
}
}

#endif // CV_HISTOGRAM_HPP
