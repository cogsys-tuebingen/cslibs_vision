#ifndef HEATMAP_HPP
#define HEATMAP_HPP

#include <opencv2/core/core.hpp>
#include <boost/function.hpp>

namespace utils_vision {
namespace heatmap {

    typedef boost::function<cv::Vec3f (float)> colorFunction;

    inline void renderHeatmap(const cv::Mat &src,
                                    cv::Mat &dst,
                                    colorFunction &color,
                              const cv::Mat &mask = cv::Mat())
    {
        if(src.channels() > 1) {
            throw std::runtime_error("Single channel matrix required for rendering!");
        }

        cv::Mat tmp;
        if(src.type() != CV_32FC1)
            src.convertTo(tmp, CV_32FC1);
        else
            tmp = src.clone();

        double min;
        double max;
        cv::minMaxLoc(tmp, &min, &max, NULL, NULL, mask);
        tmp = tmp - (float) min;
        max -= min;
        tmp = tmp / (float) max;
        dst = cv::Mat(tmp.rows, tmp.cols, CV_32FC3, cv::Scalar::all(0));
        if(mask.empty()) {
            for(int i = 0 ; i < dst.rows ; ++i) {
                for(int j = 0 ; j < dst.cols ; ++j) {
                    dst.at<cv::Vec3f>(i,j) = color(tmp.at<float>(i,j));
                }
            }
        } else {
            for(int i = 0 ; i < dst.rows ; ++i) {
                for(int j = 0 ; j < dst.cols ; ++j) {
                    if(mask.at<uchar>(i,j) != 0)
                        dst.at<cv::Vec3f>(i,j) = color(tmp.at<float>(i,j));
                }
            }
        }

        dst.convertTo(dst, CV_8UC3);
    }
};
}


#endif // HEATMAP_HPP
