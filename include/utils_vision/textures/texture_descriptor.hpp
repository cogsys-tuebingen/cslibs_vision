#ifndef TEXTURE_DESCRIPTOR_HPP
#define TEXTURE_DESCRIPTOR_HPP

#include <opencv2/opencv.hpp>
#include <math.h>

/**
 * @brief The TextureDescriptor class represents the base class for
 *        any descriptor used to encode texture information.
 */
namespace utils_vision {
class TextureDescriptor {
public:
    typedef cv::Ptr<TextureDescriptor> Ptr;

protected:
    TextureDescriptor()
    {
    }
};
}


#endif // TEXTURE_DESCRIPTOR_HPP

