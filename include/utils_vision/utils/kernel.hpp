#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <opencv2/core/core.hpp>

namespace utils_vision {
template<typename number_t>
void binomialCoefficients(std::vector<number_t>& row)
{
   if ( row.size() > 1 )
   {
      typename std::vector<number_t>::iterator r_0(row.begin()), r_i;
      typename std::vector<number_t>::reverse_iterator r_re(row.rend()-1), r_ri(r_re), c_ri;
      *r_0 = 1;
      for ( r_i = row.begin() + 1; r_i != row.end(); r_i++ )
      {
         *r_i = *r_0;
         for ( c_ri = r_ri--; c_ri != r_re; c_ri++) *c_ri += *(c_ri+1);
      }
   }
}

inline void buildBinomialKernel(cv::Mat &dst, int kernel_size)
{
    std::vector<float> binom(kernel_size);
    binomialCoefficients(binom);
    dst = cv::Mat(1, kernel_size, CV_32F, binom.data(), true);
    cv::gemm(dst, dst, 1.0 / std::pow(2.0, (kernel_size - 1) * 2.0),
             cv::Mat(), 0.0, dst, cv::GEMM_1_T);
}
}


#endif // KERNEL_HPP
