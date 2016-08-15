#ifndef WLD_HPP
#define WLD_HPP

#include <opencv2/core/core.hpp>

#define M_PI_8 0.39269908169872415481
#define SHORTENED  31.0  / M_PI
#define STANDARD   127.0 / M_PI
#define THRESHOLD_1 M_PI_8
#define THRESHOLD_2 M_PI*3.0/8.0
#define THREHSOLD_3 M_PI*5.0/8.0
#define THRESHOLD_4 M_PI*7.0/8.0

namespace cslibs_vision {
class WLD
{
public:
    WLD() = delete;

    static inline int standardRows(const int src_rows)
    {
        return src_rows - 2;
    }
    static inline int standardCols(const int src_cols)
    {
        return src_cols - 2;
    }

    static inline void standard(const cv::Mat &src,
                                cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _standard<uchar>(src, dst);  break;
        case CV_8SC1: _standard<char>(src, dst);   break;
        case CV_16UC1:_standard<ushort>(src, dst); break;
        case CV_16SC1:_standard<short>(src, dst);  break;
        case CV_32SC1:_standard<int>(src, dst);    break;
        case CV_32FC1:_standard<float>(src, dst);  break;
        case CV_64FC1:_standard<double>(src, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

    static inline int shortenedRows(const int src_rows)
    {
        return src_rows - 2;
    }
    static inline int shortenedCols(const int src_cols)
    {
        return src_cols - 2;
    }
    static inline void shortened(const cv::Mat &src,
                                 cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _shortened<uchar>(src, dst);  break;
        case CV_8SC1: _shortened<char>(src, dst);   break;
        case CV_16UC1:_shortened<ushort>(src, dst); break;
        case CV_16SC1:_shortened<short>(src, dst);  break;
        case CV_32SC1:_shortened<int>(src, dst);    break;
        case CV_32FC1:_shortened<float>(src, dst);  break;
        case CV_64FC1:_shortened<double>(src, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

    static inline int orientedRows(const int src_rows)
    {
        return src_rows - 2;
    }
    static inline int orientedCols(const int src_cols)
    {
        return src_cols - 2;
    }

    static inline void oriented(const cv::Mat &src,
                                cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _oriented<uchar>(src, dst);  break;
        case CV_8SC1: _oriented<char>(src, dst);   break;
        case CV_16UC1:_oriented<ushort>(src, dst); break;
        case CV_16SC1:_oriented<short>(src, dst);  break;
        case CV_32SC1:_oriented<int>(src, dst);    break;
        case CV_32FC1:_oriented<float>(src, dst);  break;
        case CV_64FC1:_oriented<double>(src, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

private:
    template <typename _Tp>
    static inline void _standard(const cv::Mat& src,
                                 cv::Mat& dst)
    {
        dst = cv::Mat(src.rows-2, src.cols-2, CV_8UC1, cv::Scalar());

        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar     *dst_ptr = dst.ptr<uchar>();

        double alpha=3.0, beta=0.0,diff=0.0,sigma=0.0;
        _Tp center = 0;
        int prev, pos, next = 0;
        uchar code = 0;
        int tmp_code = 0;

        for(int i=1 ; i<src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                pos  = i * src.cols + j;
                prev = pos - src.cols;
                next = pos + src.cols;

                center = src_ptr[pos];
                code = 0;
                diff = src_ptr[prev-1] + src_ptr[prev] + src_ptr[prev+1] +
                       src_ptr[pos-1] + src_ptr[pos+1] +
                       src_ptr[next-1] + src_ptr[next] + src_ptr[next+1] -
                       8.0 * center;

                sigma    = atan2(diff*alpha, beta+center);
                tmp_code = (int)((sigma+M_PI_2) * STANDARD);
                code = (unsigned char) tmp_code;
                *dst_ptr = code;
                ++dst_ptr;
            }
        }
    }

    template <typename _Tp>
    static inline void _shortened(const cv::Mat &src,
                                  cv::Mat &dst)
    {
        dst = cv::Mat(src.rows-2, src.cols-2, CV_8UC1, cv::Scalar());

        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar     *dst_ptr = dst.ptr<uchar>();

        double alpha=3.0, beta=1.0,diff=0.0,sigma=0.0;
        _Tp center = 0;
        int prev, pos, next = 0;
        uchar code = 0;
        int tmp_code = 0;

        for(int i=1;i<src.rows-1;++i) {
            for(int j=1;j<src.cols-1;++j) {
                pos  = i * src.cols + j;
                prev = pos - src.cols;
                next = pos + src.cols;

                center = src_ptr[pos];
                code = 0;

                diff = src_ptr[prev-1] + src_ptr[prev] + src_ptr[prev+1] +
                       src_ptr[pos-1] + src_ptr[pos+1] +
                       src_ptr[next-1] + src_ptr[next] + src_ptr[next+1] -
                       8.0 * center;


                sigma=atan2(diff*alpha, beta+center);
                tmp_code=(int)((sigma+M_PI_2) * SHORTENED);
                code = (unsigned char) tmp_code;
                *dst_ptr = code;
                ++dst_ptr;
            }
        }
    }

    template <typename _Tp>
    static inline void _oriented(const cv::Mat &src,
                                 cv::Mat &dst)
    {
        dst = cv::Mat(src.rows-2, src.cols-2, CV_8UC1, cv::Scalar());

        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar     *dst_ptr = dst.ptr<uchar>();
        uchar code = 0;

        double diff1, diff2, theta = 0.0;
        int prev, pos, next = 0;

        for(int i=1 ; i<src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                pos  = i * src.cols + j;
                prev = pos - src.cols;
                next = pos + src.cols;

                code  = 0;
                diff1 = src_ptr[next]-src_ptr[prev];
                diff2 = src_ptr[pos-1]-src_ptr[pos+1];
                theta=atan2(diff2, diff1);

                if(theta<THRESHOLD_1&&theta>=-THRESHOLD_1)
                    code=0;
                else if(theta<THRESHOLD_2&&theta>=THRESHOLD_1)
                    code=1;
                else if(theta<THREHSOLD_3&&theta>=THRESHOLD_2)
                    code=2;
                else if(theta<THRESHOLD_4&&theta>=THREHSOLD_3)
                    code=3;
                else if(theta<-THRESHOLD_1&&theta>=-THRESHOLD_2)
                    code=7;
                else if(theta<-THRESHOLD_2&&theta>=-THREHSOLD_3)
                    code=6;
                else if(theta<-THREHSOLD_3&&theta>=-THRESHOLD_4)
                    code=5;
                else
                    code=4;
                *dst_ptr = code;
                ++dst_ptr;
            }
        }
    }
};
}
#endif // WLD_HPP

