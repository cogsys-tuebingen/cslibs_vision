/// HEADER
#include <cslibs_vision/utils/extractor.h>

/// PROJECT
#include <cslibs_vision/common/global.hpp>

Extractor::Extractor(int)
    : keypoint("none"), descriptor("none")
{
}

void Extractor::extract(const cv::Mat& frame, const cv::Mat& mask_roi, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors) const
{
    extractKeypoints(frame, mask_roi, keypoints);
    extractDescriptors(frame, keypoints, descriptors);
}

void Extractor::extractDescriptors(const cv::Mat &frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const
{
    assert(descriptor_extractor);
    try {
        descriptor_extractor->compute(frame, keypoints, descriptors);
    } catch(cv::Exception& e) {
        ERROR("Extraction of descriptors failed");
    }
}


void Extractor::extractKeypoints(const cv::Mat& frame, const cv::Mat& mask_roi, std::vector<cv::KeyPoint> &keypoints) const
{
    if(frame.rows < 20 || frame.cols < 20) {
        WARN("frame is too small to extract features: " << frame.rows << "x" << frame.cols);
        return;
    }

//    cv::Mat gray;
//    if(frame.channels() == 1) {
//        gray = frame;
//    } else {
//        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//    }

    try {
        detector->detect(frame, keypoints, mask_roi);
    } catch(cv::Exception& e) {
        ERROR("Extraction of keypoints failed");
    }
}
