#include <opencv2/opencv.hpp>
#include <utils_vision/utils/color_functions.hpp>
#include <utils_vision/utils/stereo_parameters.hpp>

using namespace utils_vision;

#define INPUT_ERROR std::cerr << "utils_vision_test_stereo <video_file> <stereo_parameters>" << std::endl

template<typename T>
std::string toString(const T &value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

std::string getRootPath(const std::string &path)
{
    int last_slash = path.find_last_of('/');
    return path.substr(0, last_slash+1);
}

struct StereoImageSequence {
    std::string root_path;
    std::vector<std::string> paths_left;
    std::vector<std::string> paths_right;
    unsigned int size;
    unsigned int pos;
    bool loop;
    bool is_relative;

    StereoImageSequence(const bool _loop = false) :
        root_path(""),
        size(0),
        pos(0),
        loop(_loop),
        is_relative(false)
    {
    }

    void read(const std::string &path)
    {
        cv::FileStorage fsi(path, cv::FileStorage::READ);
        fsi["left"]     >> paths_left;
        fsi["right"]    >> paths_right;
        fsi["relative"] >> is_relative;
        fsi.release();

        if(is_relative)
            root_path = getRootPath(path);

        if(paths_left.size() != paths_right.size()) {
            throw std::runtime_error("Path amounts are not matching!");
        }

        size = paths_left.size();
    }

    void write(const std::string &path)
    {
        cv::FileStorage fso(path, cv::FileStorage::READ);
        fso << "left"     << paths_left;
        fso << "right"    << paths_right;
        fso << "relative" << is_relative;
        fso.release();
    }

    bool next(cv::Mat &left, cv::Mat &right)
    {
        if(pos == size)
            return false;

        left  = cv::imread(root_path + paths_left.at(pos), cv::IMREAD_COLOR);
        right = cv::imread(root_path + paths_right.at(pos), cv::IMREAD_COLOR);

        if(left.empty()) {
            throw std::runtime_error("Couldn't load left image '" + paths_left.at(pos) + "'!");
        }
        if(right.empty()) {
            throw std::runtime_error("Couldn't load right image '" + paths_right.at(pos) + "'!");
        }
        ++pos;

        if(loop && pos >= size) {
            pos = 0;
        }
    }

    void reset()
    {
        pos = 0;
    }

    bool hasNext()
    {
        return pos != size;
    }

};

inline void depthColor(const cv::Mat &src, cv::Mat &dst)
{
    assert(src.type() == CV_32FC3);

    dst = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar());

    cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>();
    const cv::Point3f *src_ptr = src.ptr<cv::Point3f>();

    cv::Mat distances(src.rows, src.cols, CV_32FC1, cv::Scalar());
    float *distances_ptr = distances.ptr<float>();

    float max = std::numeric_limits<float>::min();
    for(unsigned int i = 0 ; i < src.rows ; ++i) {
        for(unsigned int j = 0 ; j < src.cols ; ++j) {
            const cv::Point3f &pt = src_ptr[i * src.cols + j];
            float dist = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
            if(dist == std::numeric_limits<float>::infinity()) {
                dist = 0;
            }

            distances_ptr[i * src.cols + j] = dist;
            if(dist > max && dist < std::numeric_limits<float>::infinity() && dist < 10000)
                max = dist;

        }
    }

    cv::normalize(distances, distances, 0, 1, CV_MINMAX);
    cv::imshow("raw depth", distances);

    for(unsigned int i = 0; i < src.rows ; ++i) {
        for(unsigned int j = 0 ; j < src.cols ; ++j) {
            if(distances_ptr[i * src.cols + j] < std::numeric_limits<float>::infinity() &&
                    distances_ptr[i * src.cols + j] < 10000 &&
                    distances_ptr[i * src.cols + j] > 0)
                dst_ptr[i * src.cols + j] = utils_vision::color::bezierColor<cv::Vec3b>(distances_ptr[i * src.cols + j] / max);
        }
    }
}


void runSBM(StereoImageSequence &s,
            const StereoParameters &p)
{
    cv::Ptr<CvStereoBMState> parameters(cvCreateStereoBMState(CV_STEREO_BM_BASIC));
    parameters->SADWindowSize = 5;
    parameters->numberOfDisparities = 80;
    parameters->preFilterSize = 5;
    parameters->preFilterCap = 61;
    parameters->minDisparity = -39;
    parameters->textureThreshold = 507;
    parameters->uniquenessRatio = 0;
    parameters->speckleWindowSize = 0;
    parameters->speckleRange = 8;
    parameters->disp12MaxDiff = 2;

    int disparities = 4;
    int pref_filter = 5;
    int sad_window = 5;
    cv::namedWindow("parameters");
    cv::createTrackbar("preFilterSize", "parameters", &(pref_filter), 255);
    cv::createTrackbar("preFilterCap",  "parameters", &(parameters->preFilterCap), 63);
    cv::createTrackbar("SADWindowSize", "parameters", &(sad_window), 255);
    cv::createTrackbar("minDisparity",  "parameters", &(parameters->minDisparity), 255);
    cv::createTrackbar("numberOfDisparities", "parameters", &(disparities), 255);

    cv::createTrackbar("textureThreshold", "parameters", &(parameters->textureThreshold), 255);
    cv::createTrackbar("uniquenessRatio",  "parameters", &(parameters->uniquenessRatio), 255);
    cv::createTrackbar("speckleWindowSize", "parameters", &(parameters->speckleWindowSize), 255);
    cv::createTrackbar("speckleRange",  "parameters", &(parameters->speckleRange), 255);
    cv::createTrackbar("trySmallerWindows", "parameters", &(parameters->trySmallerWindows), 1);
    cv::createTrackbar("disp12MaxDiff", "parameters", &(parameters->disp12MaxDiff), 255);


    cv::Mat left;
    cv::Mat right;
    cv::Mat disparity;
    cv::Mat disparity_vis;
    cv::Mat points_3D;
    cv::Mat points_3D_vis;
    while(s.hasNext()) {
//        parameters->preFilterSize = std::max(5, parameters->preFilterSize);
//        parameters->numberOfDisparities = std::max(16, disparities * 16);
//        parameters->preFilterSize = std::max(5, pref_filter + (1 - pref_filter % 2));
        parameters->SADWindowSize = std::max(5, sad_window + (1 -  sad_window % 2));

        cv::StereoBM sbm(CV_STEREO_BM_BASIC);
        sbm.state = parameters;

        s.next(left, right);

        if(left.type() == CV_8UC3)
            cv::cvtColor(left, left, CV_BGR2GRAY);
        if(right.type() == CV_8UC3)
            cv::cvtColor(right, right, CV_BGR2GRAY);

        sbm(left, right, disparity, CV_32F);
        cv::reprojectImageTo3D(disparity,points_3D,p.Q);

        double min, max;
        cv::minMaxLoc(disparity, &min, &max);
        cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
        disparity.convertTo( disparity_vis, CV_8UC1, 255.0 );

        depthColor(points_3D, points_3D_vis);

        cv::imshow("video left", left);
        cv::imshow("video right", right);
        cv::imshow("disparity", disparity_vis);
        cv::imshow("depth", points_3D_vis);

        int key = cv::waitKey(19) & 0xFF;
        if(key == 27)
            break;
    }

}

void runSGBM(StereoImageSequence &s,
             const StereoParameters &p)
{
    cv::StereoSGBM sgbm;
//    sgbm.SADWindowSize = 5;
//    sgbm.numberOfDisparities = 192;
//    sgbm.preFilterCap = 4;
//    sgbm.minDisparity = -64;
//    sgbm.uniquenessRatio = 1;
//    sgbm.speckleWindowSize = 150;
//    sgbm.speckleRange = 2;
//    sgbm.disp12MaxDiff = 10;
//    sgbm.fullDP = false;
//    sgbm.P1 = 600;
//    sgbm.P2 = 2400;

    cv::Mat left;
    cv::Mat right;
    cv::Mat disparity;
    cv::Mat disparity_vis;
    cv::Mat points_3D;
    cv::Mat points_3D_vis;
    while(s.hasNext()) {
        s.next(left, right);

        if(left.type() == CV_8UC3)
            cv::cvtColor(left, left, CV_BGR2GRAY);
        if(right.type() == CV_8UC3)
            cv::cvtColor(right, right, CV_BGR2GRAY);

        sgbm(left, right, disparity);
        cv::reprojectImageTo3D(disparity,points_3D,p.Q);

        double min, max;
        cv::minMaxLoc(disparity, &min, &max);
        cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
        disparity.convertTo( disparity_vis, CV_8UC1, 255.0 );

        depthColor(points_3D, points_3D_vis);

        cv::imshow("video left", left);
        cv::imshow("video right", right);
        cv::imshow("disparity", disparity_vis);
        cv::imshow("depth", points_3D_vis);

        int key = cv::waitKey(19) & 0xFF;
        if(key == 27)
            break;
    }
}




int main(int argc, char *argv[])
{
    if(argc < 3) {
        INPUT_ERROR;
        return 1;
    }


    std::string path_img_seq = argv[1];
    std::string path_calib = argv[2];
    StereoParameters p;
    p.read(path_calib);
    StereoImageSequence s;
    s.read(path_img_seq);

    runSBM(s, p);
//    runSGBM(s, p);

    return 0;
}

