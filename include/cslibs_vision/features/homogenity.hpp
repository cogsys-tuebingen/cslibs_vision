#ifndef HOMOGENITY_HPP
#define HOMOGENITY_HPP

#include <opencv2/core/core.hpp>

namespace cslibs_vision {
class Homogenity
{
public:
    Homogenity() = delete;

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

    static inline int textureRows(const int src_rows)
    {
        return src_rows;
    }
    static inline int textureCols(const int src_cols)
    {
        return src_cols;
    }

    static inline void texture(const cv::Mat &src,
                               cv::Mat &dst)
    {
        switch(src.type()) {
        case CV_8UC1: _texture<uchar>(src, dst);  break;
        case CV_8SC1: _texture<char>(src, dst);   break;
        case CV_16UC1:_texture<ushort>(src, dst); break;
        case CV_16SC1:_texture<short>(src, dst);  break;
        case CV_32SC1:_texture<int>(src, dst);    break;
        case CV_32FC1:_texture<float>(src, dst);  break;
        case CV_64FC1:_texture<double>(src, dst); break;
        default: throw std::runtime_error("Unsupported matrix type!");
        }
    }

private:
    template<typename _Tp>
    static inline void _standard(const cv::Mat &src,
                                 cv::Mat &dst)
    {
        dst = cv::Mat(src.rows-2, src.cols-2, CV_8UC1, cv::Scalar());
        cv::Mat var = cv::Mat(src.rows-2, src.cols-2, CV_32FC1, cv::Scalar());
        cv::Mat gre = cv::Mat(src.rows-2, src.cols-2, CV_32FC1, cv::Scalar());

        const _Tp *src_ptr = src.ptr<_Tp>();
        uchar *dst_ptr = dst.ptr<uchar>();
        float *var_ptr = var.ptr<float>();
        float *gre_ptr = gre.ptr<float>();

        int    prev, pos, next = 0;
        int    pos_gre = 0;

        double max_var=-1.0;
        double max_gre=-1.0;
        double ave=0.0;
        double diffs[9];
        double gre_x, gre_y = 0.0;

        for(int i=1 ; i<src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                pos  = i * src.cols + j;
                prev = pos - src.cols;
                next = pos + src.cols;
                pos_gre = (i-1) * gre.cols + j-1;

                ave = (src_ptr[prev - 1] + src_ptr[prev] + src_ptr[prev + 1] +
                       src_ptr[pos  - 1] + src_ptr[pos]  + src_ptr[pos  + 1] +
                       src_ptr[next - 1] + src_ptr[next] + src_ptr[next + 1]) / 9.0;

                diffs[0] = src_ptr[prev - 1] - ave;
                diffs[1] = src_ptr[prev    ] - ave;
                diffs[2] = src_ptr[prev + 1] - ave;
                diffs[3] = src_ptr[pos  - 1] - ave;
                diffs[4] = src_ptr[pos     ] - ave;
                diffs[5] = src_ptr[pos  + 1] - ave;
                diffs[6] = src_ptr[next - 1] - ave;
                diffs[7] = src_ptr[next    ] - ave;
                diffs[8] = src_ptr[next + 1] - ave;

                var_ptr[pos_gre] = sqrt((diffs[0]*diffs[0]+
                                         diffs[1]*diffs[1]+
                                         diffs[2]*diffs[2]+
                                         diffs[3]*diffs[3]+
                                         diffs[4]*diffs[4]+
                                         diffs[5]*diffs[5]+
                                         diffs[6]*diffs[6]+
                                         diffs[7]*diffs[7]+
                                         diffs[8]*diffs[8]) / 9.0);

                if(var_ptr[pos_gre] > max_var)
                    max_var=var_ptr[pos_gre];

                gre_x =     src_ptr[prev+1] - src_ptr[prev-1] +
                        2.0*src_ptr[pos +1] - src_ptr[pos -1]*2.0 +
                            src_ptr[next+1] - src_ptr[next-1];
                gre_y =     src_ptr[next+1] - src_ptr[prev+1]+
                        2.0*src_ptr[next]   - src_ptr[prev  ]*2.0 +
                            src_ptr[next-1] - src_ptr[prev-1];
                gre_ptr[pos_gre] = hypot(gre_x, gre_y);

                if(gre_ptr[pos_gre]>max_gre)
                    max_gre=gre_ptr[pos_gre];
            }
        }

        double max_var_inv = 1.0 / max_var;
        for(int i=0; i<var.rows; ++i) {
            for(int j=0; j<var.cols; ++j) {
                pos_gre = i * var.cols + j;
                dst_ptr[pos_gre] = (1.0 - var_ptr[pos_gre] * max_var_inv) * 255.0;

            }
        }
    }

    template<typename _Tp>
    static inline void _texture(const cv::Mat &src,
                                cv::Mat &dst)
    {
        _standard<_Tp>(src, dst);
        cv::Mat inner(dst, cv::Rect(1,1,dst.cols-2, dst.rows-2));
        cv::copyMakeBorder(inner, dst, 1, 1, 1, 1, cv::BORDER_REFLECT);
    }

};
}
#endif // HOMOGENITY_HPP

