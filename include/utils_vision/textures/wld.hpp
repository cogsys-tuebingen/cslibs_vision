#ifndef WLD_HPP
#define WLD_HPP
#include "texture_descriptor.hpp"

#define M_PI_8 0.39269908169872415481
#define SHORTENED  31.0  / M_PI
#define STANDARD   127.0 / M_PI

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
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);

        double alpha=3.0;
        double belta=0.0;
        for(int i=1 ; i<src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                _Tp center = src.at<_Tp>(i,j);
                unsigned char code = 0;
                double diff=src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                       +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                       +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)-8.0*center;
                double sigma=atan2(diff*alpha, belta+center);
                int tempcode=(int)((sigma+M_PI_2) * STANDARD);
                code=(unsigned char)tempcode;
                dst.at<unsigned char>(i-1,j-1) = code;
            }
        }
    }

    template <typename _Tp>
    static inline void _shortened(const cv::Mat &src,
                                  cv::Mat &dst)
    {
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);
        double alpha=3.0;
        double belta=1.0;
        for(int i=1;i<src.rows-1;i++) {
            for(int j=1;j<src.cols-1;j++) {
                _Tp center = src.at<_Tp>(i,j);
                unsigned char code = 0;
                double diff=src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                       +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                       +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)-8.0*center;
                double sigma=atan2(diff*alpha, belta+center);
                int tempcode=(int)((sigma+M_PI_2) * SHORTENED);
                code = (unsigned char) tempcode;
                dst.at<unsigned char>(i-1,j-1) = code;
            }
        }
    }

    template <typename _Tp>
    static inline void _oriented(const cv::Mat &src,
                                 cv::Mat &dst)
    {
        dst = cv::Mat_<uchar>(src.rows-2, src.cols-2, (uchar) 0);
        const double threshold1=M_PI_8;
        const double threshold2=M_PI*3.0/8.0;
        const double threshold3=M_PI*5.0/8.0;
        const double threshold4=M_PI*7.0/8.0;
        for(int i=1 ; i<src.rows-1 ; ++i) {
            for(int j=1 ; j<src.cols-1 ; ++j) {
                unsigned char code = 0;
                double diff1=src.at<_Tp>(i+1,j)-src.at<_Tp>(i-1,j);
                double diff2=src.at<_Tp>(i,j-1)-src.at<_Tp>(i,j+1);
                double theta=atan2(diff2, diff1);
                if(theta<threshold1&&theta>=-threshold1)
                    code=0;
                else if(theta<threshold2&&theta>=threshold1)
                    code=1;
                else if(theta<threshold3&&theta>=threshold2)
                    code=2;
                else if(theta<threshold4&&theta>=threshold3)
                    code=3;
                else if(theta<-threshold1&&theta>=-threshold2)
                    code=7;
                else if(theta<-threshold2&&theta>=-threshold3)
                    code=6;
                else if(theta<-threshold3&&theta>=-threshold4)
                    code=5;
                else
                    code=4;
                dst.at<unsigned char>(i-1,j-1) = code;
            }
        }
    }
};
}
#endif // WLD_HPP

