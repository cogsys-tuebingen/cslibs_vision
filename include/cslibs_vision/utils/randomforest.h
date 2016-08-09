#ifndef CMP_RANDOMFOREST_H
#define CMP_RANDOMFOREST_H
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

namespace cslibs_vision {
class RandomForest
{
public:
    typedef cv::Ptr<RandomForest> Ptr;
    typedef cv::Ptr<cv::RandomTrees> CvForestPtr;

    /**
     * @brief RandomForest container constructor.
     */
    RandomForest();
    /**
     * @brief Load random forest from file.
     * @param path the file path
     * @return
     */
    bool load(const std::string &path);
    /**
     * @brief Check if underlying data structure is already trained.
     * @return dat structure is trained
     */
    bool isTrained();

    /**
     * @brief Simple prediction of a class label.
     * @param sample        vector to calculate the prediction for.
     * @param classID       the predicted id
     */
    void predictClass(const cv::Mat &sample, int &classID);
    void predictClassProb(const cv::Mat &sample, int &classID, float &prob);
    void predictClassProbs(const cv::Mat &sample, std::map<int, float> &probs);
    void predictClassProbs(const cv::Mat &sample, std::vector<int> &classIDs, std::vector<float> &probs);
    void predictClassProbMultiSample    (const cv::Mat &samples, int &classID, float &prob);
    void predictClassProbMultiSampleMax (const cv::Mat &samples, int &classID, float &prob);
    void predictClassProbsMultiSample   (const cv::Mat &samples, std::map<int, float> &probs);
    void predictClassProbsMultiSampleMax(const cv::Mat &samples, std::map<int, float> &probs);
protected:
    struct AccProb {
        AccProb() : prob(0), norm(1){}

        double prob;
        int    norm;
    };
    typedef std::pair<int, AccProb> AccProbEntry;
    typedef std::map<int, AccProb>  AccProbIndex;


    CvForestPtr          forest_;
    std::vector<float>   priors_;
    bool                 is_trained_;

    void prediction         (const cv::Mat &sample, std::map<int, float> &probs, int &maxClassID);
};
}
#endif // CMP_RANDOMFOREST_H
