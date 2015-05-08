#ifndef COLOR_FUNCTIONS_HPP
#define COLOR_FUNCTIONS_HPP

#include <opencv2/core/core.hpp>

namespace utils_vision {
namespace color {
static const cv::Point3f red(0.f,0.f,255.f);  /// P0
static const cv::Point3f green(0.f,255.f,0.f);  /// P1
static const cv::Point3f blue(255.f,0.f,0.f);  /// p3
static const cv::Point3f fac1  (blue - 2 * green + red);
static const cv::Point3f fac2  (-2*blue + 2 * green);

template<typename VecT>
inline VecT bezierColor(const float value)
{
    cv::Point3f  col = fac1 * value * value + fac2 * value + blue;
    return VecT(std::floor(col.x + .5), std::floor(col.y + .5), std::floor(col.z + .5));
}

template<typename VecT>
inline VecT parabolaColor(const float value)
{
    cv::Point3f col = value * value * red + (value - 1) * (value - 1) * blue;
    return VecT(std::floor(col.x + .5), std::floor(col.y + .5), std::floor(col.z + .5));
}
}
}

#endif // COLOR_FUNCTIONS_HPP
