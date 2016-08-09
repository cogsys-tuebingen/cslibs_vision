#ifndef LBP_HPP_
#define LBP_HPP_

//! \author philipp <bytefish[at]gmx[dot]de>
//! \copyright BSD, see LICENSE.

#include "opencv2/opencv.hpp"
#include <limits>

using namespace cv;
using namespace std;

namespace lbp {

// templated functions
template <typename _Tp>
void OLBP_(const cv::Mat& src, cv::Mat& dst);           /// done

template <typename _Tp>
void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);            /// done

template <typename _Tp>
void VARLBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);            /// done

template <typename _Tp>
void OLTP_(const cv::Mat& src, cv::Mat& dst);//LTP            /// done

template <typename _Tp>
void OCSLBP_(const cv::Mat& src, cv::Mat& dst);//CS_LBP      /// done

template <typename _Tp>
void OWLD_(const cv::Mat& src, cv::Mat& dst);//WLD          /// done

template <typename _Tp>
void OWLDSHORT_(const cv::Mat& src, cv::Mat& dst);//WLD_Short /// done

template <typename _Tp>
void OWLDORI_(const cv::Mat& src, cv::Mat& dst);//WLD_Orientation   /// done

template <typename _Tp>
void OHOMOGENEITY_(const cv::Mat& src, cv::Mat& dst);//Local Homogeneity    /// done

template <typename _Tp>
void OHOMOGENEITYOFTEXTURE_(const cv::Mat& src, cv::Mat& dst);//Local Homogeneity of Texture

// wrapper functions
void OLBP(const Mat& src, Mat& dst);
void ELBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);
void VARLBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);

void OLTP(const Mat& src, Mat& dst);//LTP

void OCSLBP(const Mat& src, Mat& dst);//CSLBP

void OWLD(const Mat& src, Mat& dst);//WLD

void OWLDSHORT(const Mat& src, Mat& dst);//WLD_Short

void OWLDORI(const Mat& src, Mat& dst);//WLD_Orientation

void OHOMOGENEITY(const Mat& src, Mat& dst);//Local Homogeneity

void OHOMOGENEITYOFTEXTURE(const Mat& src, Mat& dst);//Local Homogeneity of Texture

// Mat return type functions
Mat OLBP(const Mat& src);
Mat ELBP(const Mat& src, int radius = 1, int neighbors = 8);
Mat VARLBP(const Mat& src, int radius = 1, int neighbors = 8);

}
#endif
