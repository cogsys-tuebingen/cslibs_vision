#ifndef CLUSTER_BOUNDARIES_HPP
#define CLUSTER_BOUNDARIES_HPP

#include <opencv2/opencv.hpp>

namespace cslibs_vision {
template<typename LabelT>
void getClusterBoundaryMask(const cv::Mat& src,
                            cv::Mat& dst)
{
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    const LabelT *src_ptr = src.ptr<LabelT>();
    uchar *dst_ptr = dst.ptr<uchar>();

    int pos = 0;
    for(int i = 0 ; i < src.rows; ++i) {
        for(int j = 0 ; j < src.cols ; ++j) {
            pos = i * src.cols + j;

            bool is_border = false;
            if(i > 0)
                is_border |= src_ptr[pos] != src_ptr[pos - src.cols];
            if(i < src.rows - 1)
                is_border |= src_ptr[pos] != src_ptr[pos + src.cols];
            if(j > 0)
                is_border |= src_ptr[pos] != src_ptr[pos - 1];
            if(j < src.cols - 1)
                is_border |= src_ptr[pos] != src_ptr[pos + 1];

            if(is_border)
                dst_ptr[pos] = 1;
        }
    }
}
}
#endif // CLUSTER_BOUNDARIES_HPP

