/// HEADER
#include <utils_vision/data/pose.h>

Pose::Pose()
{
}

Pose::Pose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation)
    : position(position), orientation(orientation)
{
}
