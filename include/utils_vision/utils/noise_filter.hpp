#ifndef NOISE_FILTER_HPP
#define NOISE_FILTER_HPP

#include <opencv2/core/core.hpp>

namespace utils_vision {

template<typename T, typename S>
struct ThresholdNoiseFilter {
    inline static void interpolate(const cv::Mat &src, const cv::Mat &thresholds, const S threshold,
                                   cv::Mat &dst)
    {
        assert(src.channels()    == 1);

        static int xs[] = {-1,-1,-1, 0, 1, 1, 1, 0};
        static int ys[] = {-1, 0, 1, 1, 1, 0,-1,-1};

        dst = src.clone();

        const T *src_ptr = src.ptr<T>();
        T       *dst_ptr = dst.ptr<T>();
        const S *th_ptr = thresholds.ptr<S>();

        for(int y = 0 ; y < src.rows ; ++y) {
            for(int x = 0 ; x < src.cols ; ++x) {
                int pos = y * src.cols + x;
                if(th_ptr[pos] > threshold) {
                    double normalizer = 0;
                    T      inter_value = 0;
                    for(int i = 0 ; i < 8 ; ++i) {
                        int posx = x + xs[i];
                        int posy = y + ys[i];
                        if(posx > -1 && posx < src.cols &&
                                posy > -1 && posy < src.rows) {
                            int npos = posy * src.cols + posx;
                            if(th_ptr[npos] <= threshold && src_ptr[npos] > 0) {
                                inter_value      += src_ptr[npos];
                                normalizer += 1.0;
                            }
                        }
                    }
                    dst_ptr[pos] = inter_value / normalizer;
                }
            }
        }
    }

    inline static void filter(const cv::Mat &src, const cv::Mat &thresholds, const S threshold,
                              cv::Mat &dst)
    {
        assert(src.channels()    == 1);
        dst = src.clone();

        T       *dst_ptr = dst.ptr<T>();
        const S *th_ptr = thresholds.ptr<S>();

        for(int y = 0 ; y < src.rows ; ++y) {
            for(int x = 0 ; x < src.cols ; ++x) {
                int pos = y * src.cols + x;
                if(th_ptr[pos] > threshold) {
                    dst_ptr[pos] = 0;
                }
            }
        }
    }
};


}
#endif // NOISE_FILTER_HPP

