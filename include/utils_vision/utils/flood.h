#ifndef FLOOD_H
#define FLOOD_H
#include <opencv2/core/core.hpp>
namespace utils_vision {

const unsigned short FLOOD_LABEL_DEFAULT = 0;
const unsigned short FLOOD_LABEL_START   = 1;
const unsigned short FLOOD_LABEL_IMPLODE = std::numeric_limits<unsigned short>::max();
const unsigned char  FLOOD_MASK_DEFAULT  = 0;
const unsigned char  FLOOD_MASK_SET      = 255;

void    flood(const cv::Mat &edges, const cv::Point &anchor,
              const unsigned short label, cv::Mat &labels,
              const uchar edge);

void    flood(const cv::Mat &edges, const cv::Point &anchor,
              const unsigned short label, cv::Mat &labels,
              const uchar edge, const unsigned int threshold);

void    label(const cv::Mat &edges, cv::Mat &labels,
              const uchar edge);

void    label(const cv::Mat &edges, cv::Mat &labels,
              const uchar edge, const unsigned int threshold);

inline cv::Mat labels(const cv::Mat &edges)
{
    return cv::Mat(edges.rows, edges.cols, CV_16UC1, cv::Scalar::all(FLOOD_LABEL_DEFAULT));
}
}
#endif // FLOOD_H
