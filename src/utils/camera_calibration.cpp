#include <utils_vision/utils/camera_calibration.h>
#include <stdexcept>

using namespace utils_vision;

CameraCalibration::CameraCalibration(const Mode mode, const cv::Size &board_size, const double square_size,
                                     const int kernel_size, const int flag_corner, const int flag_calib) :
    mode_(mode),
    size_board_(board_size),
    size_frame_(0,0),
    size_square_(square_size),
    enhance_kernel_(kernel_size,kernel_size),
    enhance_zero_(-1,-1),
    enhance_term_(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1),
    flag_calib_(flag_calib),
    flag_corner_(flag_corner),
    buffer_found_(false)

{
}

CameraCalibration::~CameraCalibration()
{
}

void CameraCalibration::analyze(const cv::Mat &frame)
{
    cv::Size size_frame(frame.cols, frame.rows);
    if(size_frame_ == cv::Size(0,0)) {
        size_frame_ = size_frame;
    }

    if(size_frame != size_frame_) {
        throw std::runtime_error("Cannot add frame to data set! Sizes not fitting!");
    }

    /// PREPARARTION
    buffer_corners_.clear();
    buffer_found_ = false;

    /// COMPUTATION
    switch(mode_) {
    case CHESSBOARD:
        buffer_found_ = cv::findChessboardCorners(frame, size_board_, buffer_corners_, flag_corner_);
//////// TWO WORKING SOLUTION TO THE CORNER FLAGS ///////////////////////////////////////////
//      CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
//      CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
////////
        if(buffer_found_)
            enhanceCorners(frame, buffer_corners_);
        break;
    case CIRCLES_GRID:
        buffer_found_ = cv::findCirclesGrid( frame, size_board_, buffer_corners_ );
        break;
    case ASYMMETRIC_CIRCLES_GRID:
        buffer_found_ = cv::findCirclesGrid( frame, size_board_, buffer_corners_, cv::CALIB_CB_ASYMMETRIC_GRID );
        break;
    default:
        buffer_found_ = false;
        break;
    }
}

void CameraCalibration::addFrame()
{
    if(buffer_found_) {
        buffer_image_points_.push_back(buffer_corners_);
    }
}

void CameraCalibration::drawFoundCorners(cv::Mat &frame)
{
    cv::drawChessboardCorners(frame, size_board_, cv::Mat(buffer_corners_), buffer_found_ );
}

void CameraCalibration::calibrate(cv::Mat &camera_matrix, cv::Mat &distortion)
{
    if(buffer_image_points_.empty()) {
        std::cerr << "No aquired frames so far!" << std::endl;
        return;
    }

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    camera_matrix = cv::Mat::eye(3,3,CV_64F);
    distortion    = cv::Mat::zeros(8,1,CV_64FC1);

    std::vector<std::vector<cv::Point3f> > object_points(1);
    calculateObjectPoints(object_points.at(0));
    object_points.resize(buffer_image_points_.size(), object_points.at(0));

    cv::calibrateCamera(object_points, buffer_image_points_, size_frame_,
                        camera_matrix, distortion, rvecs, tvecs, flag_calib_);
    //// CV_CALIB_FIX_K4|CV_CALIB_FIX_K5

}

void CameraCalibration::calibrateFF(cv::Mat &camera_matrix, cv::Mat &distortion, double &rms, double &average_error,
                                    cv::Mat &reprojection_errors, cv::Mat &extrinsics)
{
    if(buffer_image_points_.empty()) {
        std::cerr << "No aquired frames so far!" << std::endl;
        return;
    }

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    std::vector<float>   reprojection_err;

    camera_matrix = cv::Mat::eye(3,3,CV_64F);
    distortion    = cv::Mat::zeros(8,1,CV_64FC1);

    std::vector<std::vector<cv::Point3f> > object_points(1);
    calculateObjectPoints(object_points.at(0));
    object_points.resize(buffer_image_points_.size(), object_points.at(0));

    rms = cv::calibrateCamera(object_points, buffer_image_points_, size_frame_,
                              camera_matrix, distortion, rvecs, tvecs, flag_calib_);

    calculateReprojectionError(object_points, rvecs, tvecs, camera_matrix, distortion, reprojection_err, average_error);

    reprojection_errors = cv::Mat(reprojection_err);

    if(!rvecs.empty() && !tvecs.empty()) {
        extrinsics = cv::Mat(rvecs.size(), 6, rvecs.at(0).type());
        for(int i = 0 ; i < (int) rvecs.size() ; ++i) {
            cv::Mat r = extrinsics(cv::Range(i, i+1), cv::Range(0,3));
            cv::Mat t = extrinsics(cv::Range(i, i+1), cv::Range(3,6));

            r = rvecs.at(i).t();
            t = tvecs.at(i).t();
        }
    }
}

unsigned int CameraCalibration::sizeDataset()
{
    return buffer_image_points_.size();
}

void CameraCalibration::reset(const Mode mode, const cv::Size &board_size, const double square_size)
{
    mode_         = mode;
    size_board_   = board_size;
    size_frame_   = cv::Size(0,0);
    size_square_  = square_size;
    buffer_found_ = false;
    buffer_image_points_.clear();
}

void CameraCalibration::enhanceCorners(const cv::Mat &frame, std::vector<cv::Point2f> &corners)
{
    cv::Mat gray;
    if(frame.type() == CV_8UC1)
        gray = frame;
    else
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
    cv::cornerSubPix(gray, corners, enhance_kernel_, enhance_zero_, enhance_term_);
}

void CameraCalibration::calculateObjectPoints(std::vector<cv::Point3f> &object_points)
{
    switch(mode_) {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for(int i = 0 ; i < size_board_.height ; ++i)
            for(int j = 0 ; j < size_board_.width ; ++j)
                object_points.push_back(cv::Point3f(size_square_ * j, size_square_ * i , 0.f));
        break;
    case ASYMMETRIC_CIRCLES_GRID:
        for(int i = 0 ; i < size_board_.height ; ++i)
            for(int j = 0 ; j < size_board_.width ; ++j)
                object_points.push_back(cv::Point3f(size_square_ * (2*j + i % 2), size_square_ * i , 0.f));
        break;
    default:
        break;
    }
}

void CameraCalibration::calculateReprojectionError(const std::vector<std::vector<cv::Point3f> > &object_points, const std::vector<cv::Mat> &rvecs,
                                                   const std::vector<cv::Mat> &tvecs, cv::Mat &camera_matrix, const cv::Mat &distortion,
                                                   std::vector<float> &reprojection_errors, double &average_error)
{
    std::vector<cv::Point2f> image_points;
    int total_points(0);
    unsigned int i(0);
    double total_error(0.0);
    for(std::vector<std::vector<cv::Point3f> >::const_iterator it = object_points.begin() ; it != object_points.end() ; ++it, ++i) {
        cv::projectPoints(cv::Mat(*it), rvecs.at(i), tvecs.at(i), camera_matrix, distortion, image_points);
        double error = cv::norm(cv::Mat(buffer_image_points_.at(i)), cv::Mat(image_points), CV_L2);
        error *= error;
        unsigned int n = (*it).size();
        reprojection_errors.push_back(std::sqrt(error / (double) n));
        total_error  += error;
        total_points += n;
    }
    average_error = std::sqrt(total_error / (double) total_points);
}
