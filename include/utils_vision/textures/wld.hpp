#ifndef WLD_HPP
#define WLD_HPP
#include "texture_descriptor.hpp"

namespace utils_vision {
class WLD : public TextureDescriptor
{
public:
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


private:
    template <typename _Tp>
    void _standard(const cv::Mat& src,
                   cv::Mat& dst)
    {
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);

        double alpha=3.0;
        double belta=0.0;
        double pi=3.141592653589;
        for(int i=1;i<src.rows-1;i++) {
            for(int j=1;j<src.cols-1;j++) {
                _Tp center = src.at<_Tp>(i,j);
                unsigned char code = 0;
                double diff=src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                       +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                       +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)-8.0*center;
                double sigma=atan2(diff*alpha, belta+center);
                int tempcode=(int)((sigma+pi/2)*127.0/pi);
                code=(unsigned char)tempcode;
                dst.at<unsigned char>(i-1,j-1) = code;
            }
        }
    }
};
}
#endif // WLD_HPP

