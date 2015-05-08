#ifndef CLUSTER_BOUNDARIES_HPP
#define CLUSTER_BOUNDARIES_HPP

#include <opencv2/opencv.hpp>

namespace utils_vision {
template<typename LabelT>
void getClusterBoundaryMask(const cv::Mat& src,
                            cv::Mat& dst)
{
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    const LabelT *src_ptr = src.ptr<LabelT>();
    uchar *dst_ptr = dst.ptr<uchar>();
    for(int i = 1 ; i < src.rows - 1 ; ++i) {
        for(int j = 1 ; j < src.cols - 1; ++j) {
            int pos = i * src.cols + j;
            if(src_ptr[pos] != src_ptr[pos + src.cols] || src_ptr[pos] != src_ptr[pos + 1]) {
                dst_ptr[pos] = 1;
            }
        }
    }
}
}
#endif // CLUSTER_BOUNDARIES_HPP

