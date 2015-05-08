#include <utils_vision/utils/flood.h>

#include <deque>
#include <opencv2/imgproc/imgproc.hpp>
namespace utils_vision {
void flood(const cv::Mat &edges, const cv::Point &anchor,
           const unsigned short label,
           cv::Mat &labels, const uchar edge)
{
    assert(label > 0);
    assert(edges.type()  == CV_8UC1);
    assert(labels.type() == CV_16UC1);
    assert(edges.rows == labels.rows);
    assert(edges.cols == labels.cols);

    if(edges.at<uchar>(anchor.y, anchor.x) == edge)
        return;

    std::deque<cv::Point> Q;
    Q.push_back(anchor);

    const static int N_X[] = {-1, 0, 0, 1};
    const static int N_Y[] = { 0,-1, 1, 0};

    while(!Q.empty()) {
        cv::Point ca = Q.front();
        Q.pop_front();

        /// ALREADY VISITED
        if(labels.at<unsigned short>(ca.y, ca.x) != FLOOD_LABEL_DEFAULT)
            continue;

        /// NOT ALREADY VISITED
        labels.at<unsigned short>(ca.y, ca.x) = label;

        for(int n = 0 ; n < 4 ; ++n) {
            cv::Point neighbour(ca.x + N_X[n], ca.y + N_Y[n]);
            if(neighbour.x < 0 || neighbour.x >= edges.cols ||
                    neighbour.y < 0 || neighbour.y >= edges.rows)
                continue;
            if(edges.at<uchar>(neighbour.y, neighbour.x) != edge) {
                Q.push_back(neighbour);
            }
        }
    }
}

void flood(const cv::Mat &edges, const cv::Point &anchor,
           const unsigned short label, cv::Mat &labels,
           const uchar edge, const unsigned int threshold)
{
    assert(label > 0);
    assert(edges.type()  == CV_8UC1);
    assert(labels.type() == CV_16UC1);
    assert(edges.rows == labels.rows);
    assert(edges.cols == labels.cols);

    cv::Mat mask(edges.rows, edges.cols, CV_8UC1, cv::Scalar::all(FLOOD_MASK_DEFAULT));

    if(edges.at<uchar>(anchor.y, anchor.x) == edge)
        return;

    std::deque<cv::Point> Q;
    Q.push_back(anchor);

    const static int N_X[] = {-1, 0, 0, 1};
    const static int N_Y[] = { 0,-1, 1, 0};
    unsigned int size = 0;

    while(!Q.empty()) {
        cv::Point ca = Q.front();
        Q.pop_front();

        /// ALREADY VISITED

        if(labels.at<unsigned short>(ca.y, ca.x) != FLOOD_LABEL_DEFAULT ||
                mask.at<uchar>(ca.y, ca.x) != FLOOD_MASK_DEFAULT)
            continue;

        /// NOT ALREADY VISITED
        mask.at<uchar>(ca.y, ca.x) = FLOOD_MASK_SET;
        ++size;

        for(int n = 0 ; n < 4 ; ++n) {
            cv::Point neighbour(ca.x + N_X[n], ca.y + N_Y[n]);
            if(neighbour.x < 0 || neighbour.x >= edges.cols ||
               neighbour.y < 0 || neighbour.y >= edges.rows)
                continue;

            if(edges.at<uchar>(neighbour.y, neighbour.x) != edge) {
                Q.push_back(neighbour);
            }
        }
    }

    if(size > threshold) {
        labels.setTo(cv::Scalar::all(label), mask);
    } else {
        labels.setTo(cv::Scalar::all(FLOOD_LABEL_IMPLODE), mask);
    }
}

void label(const cv::Mat &edges, cv::Mat &labels, const uchar edge)
{
    assert(edges.type() == CV_8UC1);
    unsigned short label  = FLOOD_LABEL_START;
    labels = utils_vision::labels(edges);

    const uchar*  ptr_edge = edges.ptr<uchar>();
    ushort* ptr_lab  = labels.ptr<ushort>();
    for(int y = 0 ; y < edges.rows ; ++y) {
        for(int x = 0 ; x < edges.cols ; ++x) {
            if(*ptr_edge != edge &&
                    *ptr_lab  == FLOOD_LABEL_DEFAULT) {
                cv::Point anchor(x,y);
                utils_vision::flood(edges, anchor, label, labels, edge);
                ++label;
            }
            ++ptr_edge;
            ++ptr_lab;
        }
    }
}

void label(const cv::Mat &edges, cv::Mat &labels, const uchar edge, const unsigned int threshold)
{
    assert(edges.type() == CV_8UC1);
    unsigned short label  = FLOOD_LABEL_START;
    labels = utils_vision::labels(edges);

    const uchar*  ptr_edge = edges.ptr<uchar>();
    ushort* ptr_lab  = labels.ptr<ushort>();
    for(int y = 0 ; y < edges.rows ; ++y) {
        for(int x = 0 ; x < edges.cols ; ++x) {
            if(*ptr_edge != edge &&
                    *ptr_lab  == FLOOD_LABEL_DEFAULT) {
                cv::Point anchor(x,y);
                utils_vision::flood(edges, anchor, label, labels, edge, threshold);
                ++label;
            }
            ++ptr_edge;
            ++ptr_lab;
        }
    }

    unsigned short *ptr = labels.ptr<unsigned short>();
    for(int i = 0 ; i < labels.rows * labels.cols ; ++i, ++ptr) {
        if(*ptr == FLOOD_LABEL_IMPLODE)
            *ptr = FLOOD_LABEL_DEFAULT;
    }
}
}
