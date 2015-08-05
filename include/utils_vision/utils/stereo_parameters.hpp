#ifndef STEREO_PARAMETERS_HPP
#define STEREO_PARAMETERS_HPP

#include <opencv2/core/core.hpp>

namespace utils_vision {
struct StereoParameters {
    /// the camera projection matrices
    cv::Mat CM1, CM2;
    /// the distortion coefficients
    cv::Mat D1, D2;
    /// Rotation, Translation, ...
    cv::Mat R, T, E, F;
    ///
    cv::Mat R1, R2, P1, P2, Q;

    void write(const std::string &calibration) {
        cv::FileStorage fso(calibration, cv::FileStorage::WRITE);
        fso << "CM1" << CM1;
        fso << "CM2" << CM2;
        fso << "D1"  << D1;
        fso << "D2"  << D2;
        fso << "R"   << R;
        fso << "T"   << T;
        fso << "E"   << E;
        fso << "F"   << F;
        fso << "R1"  << R1;
        fso << "R2"  << R2;
        fso << "P1"  << P1;
        fso << "P2"  << P2;
        fso << "Q"   << Q;
        fso.release();
    }
    void read(const std::string &calibration) {
        cv::FileStorage fsi(calibration, cv::FileStorage::READ);
        fsi["CM1"] >> CM1;
        fsi["CM2"] >> CM2;
        fsi["D1"]  >> D1;
        fsi["D2"]  >> D2;
        fsi["R"]   >> R;
        fsi["T"]   >> T;
        fsi["E"]   >> E;
        fsi["F"]   >> F;
        fsi["R1"]  >> R1;
        fsi["R2"]  >> R2;
        fsi["P1"]  >> P1;
        fsi["P2"]  >> P2;
        fsi["Q"]   >> Q;
        fsi.release();
    }
};
}
#endif // STEREO_PARAMETERS_HPP

