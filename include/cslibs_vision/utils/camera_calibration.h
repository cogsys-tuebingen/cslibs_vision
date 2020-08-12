#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

/// SYSTEM
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace cslibs_vision {
class CameraCalibration {
public:
    typedef boost::shared_ptr<CameraCalibration> Ptr;

    enum Mode {CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID};

    CameraCalibration(const Mode mode = CHESSBOARD,
                      const cv::Size &board_size = cv::Size(5,8),
                      const double square_size = 0.1, const int kernel_size = 11,
                      const int flag_corner = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS,
                      const int flag_calib = cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);

    virtual ~CameraCalibration();

    void analyze(const cv::Mat &frame);

    void addFrame();

    void drawFoundCorners(cv::Mat &frame);

    void calibrate(cv::Mat &camera_matrix, cv::Mat &distortion);

    /**
     * @brief calibrateFF (FULL FEADBACK)
     * @param camera_matrix
     * @param distortion
     * @param rms
     * @param average_error
     * @param reprojection_errors
     * @param extrinsics
     */
    void calibrateFF(cv::Mat &camera_matrix, cv::Mat &distortion, double &rms, double &average_error,
                     cv::Mat &reprojection_errors, cv::Mat &extrinsics);

    unsigned int sizeDataset();

    void reset(const Mode mode = CHESSBOARD,
               const cv::Size &board_size = cv::Size(5,8),
               const double square_size = 0.1);

private:
    /// SETTINGS
    Mode                mode_;
    cv::Size            size_board_;
    cv::Size            size_frame_;
    float               size_square_;

    cv::Size            enhance_kernel_;
    cv::Size            enhance_zero_;
    cv::TermCriteria    enhance_term_;
    int                 flag_calib_;
    int                 flag_corner_;


    bool                                    buffer_found_;
    std::vector<cv::Point2f>                buffer_corners_;
    std::vector<std::vector<cv::Point2f> >  buffer_image_points_;

    void enhanceCorners(const cv::Mat &frame, std::vector<cv::Point2f> &corners);
    void calculateObjectPoints(std::vector<cv::Point3f> &object_points);
    void calculateReprojectionError(const std::vector<std::vector<cv::Point3f> > &object_points, const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs,
                                    cv::Mat &camera_matrix, const cv::Mat &distortion,
                                    std::vector<float> &reprojection_errors, double &average_error);
};
}
#endif // CAMERA_CALIBRATION_H
