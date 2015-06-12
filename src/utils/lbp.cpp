#include <utils_vision/utils/lbp.h>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

template <typename _Tp>
void lbp::OLBP_(const Mat& src, Mat& dst) {
	dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
	for(int i=1;i<src.rows-1;i++) {
		for(int j=1;j<src.cols-1;j++) {
			_Tp center = src.at<_Tp>(i,j);
			unsigned char code = 0;
			code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
			code |= (src.at<_Tp>(i-1,j) > center) << 6;
			code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
			code |= (src.at<_Tp>(i,j+1) > center) << 4;
			code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
			code |= (src.at<_Tp>(i+1,j) > center) << 2;
			code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
			code |= (src.at<_Tp>(i,j-1) > center) << 0;
			dst.at<unsigned char>(i-1,j-1) = code;
		}
	}
}

//WLD
template <typename _Tp>
void lbp::OWLD_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    double alpha=3.0;
    double belta=0.0;
    double pi=3.141592653589;
    //double max=-1;
    //double min=257;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            double diff=src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                   +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                   +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)-8.0*center;
            double sigma=atan2(diff*alpha, belta+center);
            //cout<<sigma<<" ";
            int tempcode=(int)((sigma+pi/2)*127.0/pi);
            /*if(tempcode>max)
                max=tempcode;
            if(tempcode<min)
                min=tempcode;
            cout<<tempcode<<" ";*/
            code=(unsigned char)tempcode;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
    //cout<<max<<" "<<min<<std::endl;
}

//WLD_Short
template <typename _Tp>
void lbp::OWLDSHORT_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    double alpha=3.0;
    double belta=1.0;
    double pi=3.141592653589;
    //double max=-1;
    //double min=257;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            double diff=src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                   +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                   +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)-8.0*center;
            double sigma=atan2(diff*alpha, belta+center);
            //cout<<sigma<<" ";
            int tempcode=(int)((sigma+pi/2)*31.0/pi);
            /*if(tempcode>max)
                max=tempcode;
            if(tempcode<min)
                min=tempcode;
            cout<<tempcode<<" ";*/
            code=(unsigned char)tempcode;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
    //cout<<max<<" "<<min<<std::endl;
}

//WLD_Orientation
template <typename _Tp>
void lbp::OWLDORI_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    double pi=3.141592653589;
    double threshold1=pi/8.0;
    double threshold2=pi*3.0/8.0;
    double threshold3=pi*5.0/8.0;
    double threshold4=pi*7.0/8.0;
    //double max=-1;
    //double min=257;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            unsigned char code = 0;
            double diff1=src.at<_Tp>(i+1,j)-src.at<_Tp>(i-1,j);
            double diff2=src.at<_Tp>(i,j-1)-src.at<_Tp>(i,j+1);
            //src.at<_Tp>(i-1,j-1)++src.at<_Tp>(i-1,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j-1)
            double theta=atan2(diff2, diff1);
            //cout<<theta<<" ";
            if(theta<threshold1&&theta>=-threshold1)
                code=0;
            else if(theta<threshold2&&theta>=threshold1)
                code=1;
            else if(theta<threshold3&&theta>=threshold2)
                code=2;
            else if(theta<threshold4&&theta>=threshold3)
                code=3;
            else if(theta<-threshold1&&theta>=-threshold2)
                code=7;
            else if(theta<-threshold2&&theta>=-threshold3)
                code=6;
            else if(theta<-threshold3&&theta>=-threshold4)
                code=5;
            else
                code=4;
            //cout<<(int)code<<" ";
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
    //cout<<max<<" "<<min<<std::endl;
}

//Local Homogeneity
template <typename _Tp>
void lbp::OHOMOGENEITY_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    Mat var, gre;
    var = Mat::zeros(src.rows-2, src.cols-2, CV_32F);
    gre = Mat::zeros(src.rows-2, src.cols-2, CV_32F);
    double max_var=-1.0;
    double max_gre=-1.0;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            double ave=(src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                   +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                   +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)+src.at<_Tp>(i,j))/9.0;
            var.at<float>(i-1,j-1)=sqrt(((src.at<_Tp>(i-1,j-1)-ave)*(src.at<_Tp>(i-1,j-1)-ave)+
                                         (src.at<_Tp>(i-1,j)-ave)*(src.at<_Tp>(i-1,j)-ave)+
                                         (src.at<_Tp>(i-1,j+1)-ave)*(src.at<_Tp>(i-1,j+1)-ave)+
                                         (src.at<_Tp>(i,j+1)-ave)*(src.at<_Tp>(i,j+1)-ave)+
                                         (src.at<_Tp>(i+1,j+1)-ave)*(src.at<_Tp>(i+1,j+1)-ave)+
                                         (src.at<_Tp>(i+1,j)-ave)*(src.at<_Tp>(i+1,j)-ave)+
                                         (src.at<_Tp>(i+1,j-1)-ave)*(src.at<_Tp>(i+1,j-1)-ave)+
                                         (src.at<_Tp>(i,j-1)-ave)*(src.at<_Tp>(i,j-1)-ave)+
                                         (src.at<_Tp>(i,j)-ave)*(src.at<_Tp>(i,j)-ave))/9.0);
            if(var.at<float>(i-1,j-1)>max_var)
                max_var=var.at<float>(i-1,j-1);
            double gre_x = src.at<_Tp>(i-1,j+1) - src.at<_Tp>(i-1,j-1) +
                           src.at<_Tp>(i,j+1)*2 - src.at<_Tp>(i,j-1)*2 +
                           src.at<_Tp>(i+1,j+1) - src.at<_Tp>(i+1,j-1);
            double gre_y=  src.at<_Tp>(i+1,j-1) - src.at<_Tp>(i-1,j-1) +
                           src.at<_Tp>(i+1,j)*2 - src.at<_Tp>(i-1,j)*2 +
                           src.at<_Tp>(i+1,j+1) - src.at<_Tp>(i-1,j+1);
            gre.at<float>(i-1,j-1)=sqrt(gre_x*gre_x+gre_y*gre_y);
            if(gre.at<float>(i-1,j-1)>max_gre)
                max_gre=gre.at<float>(i-1,j-1);
        }
    }
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            //double tempvalue = 1.0-(var.at<float>(i-1,j-1)/max_var+gre.at<float>(i-1,j-1)/max_gre)/2.0;
            //double tempvalue = 1.0-(var.at<float>(i-1,j-1)/max_var*gre.at<float>(i-1,j-1)/max_gre);
            double tempvalue = 1.0-var.at<float>(i-1,j-1)/max_var;
            dst.at<unsigned char>(i-1,j-1) = (unsigned char) (tempvalue*255.0);
        }
    }
    //cout<<max<<" "<<min<<std::endl;
}

//Local Homogeneity of Texture
template <typename _Tp>
void lbp::OHOMOGENEITYOFTEXTURE_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat var, gre;
    var = Mat::zeros(src.rows-2, src.cols-2, CV_32F);
    gre = Mat::zeros(src.rows-2, src.cols-2, CV_32F);
    double max_var=-1.0;
    double max_gre=-1.0;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            double ave=(src.at<_Tp>(i-1,j-1)+src.at<_Tp>(i-1,j)+src.at<_Tp>(i-1,j+1)
                   +src.at<_Tp>(i,j+1)+src.at<_Tp>(i+1,j+1)+src.at<_Tp>(i+1,j)
                   +src.at<_Tp>(i+1,j-1)+src.at<_Tp>(i,j-1)+src.at<_Tp>(i,j))/9.0;
            var.at<float>(i-1,j-1)=sqrt(((src.at<_Tp>(i-1,j-1)-ave)*(src.at<_Tp>(i-1,j-1)-ave)+
                                         (src.at<_Tp>(i-1,j)-ave)*(src.at<_Tp>(i-1,j)-ave)+
                                         (src.at<_Tp>(i-1,j+1)-ave)*(src.at<_Tp>(i-1,j+1)-ave)+
                                         (src.at<_Tp>(i,j+1)-ave)*(src.at<_Tp>(i,j+1)-ave)+
                                         (src.at<_Tp>(i+1,j+1)-ave)*(src.at<_Tp>(i+1,j+1)-ave)+
                                         (src.at<_Tp>(i+1,j)-ave)*(src.at<_Tp>(i+1,j)-ave)+
                                         (src.at<_Tp>(i+1,j-1)-ave)*(src.at<_Tp>(i+1,j-1)-ave)+
                                         (src.at<_Tp>(i,j-1)-ave)*(src.at<_Tp>(i,j-1)-ave)+
                                         (src.at<_Tp>(i,j)-ave)*(src.at<_Tp>(i,j)-ave))/9.0);
            if(var.at<float>(i-1,j-1)>max_var)
                max_var=var.at<float>(i-1,j-1);
            double gre_x = src.at<_Tp>(i-1,j+1) - src.at<_Tp>(i-1,j-1) +
                           src.at<_Tp>(i,j+1)*2 - src.at<_Tp>(i,j-1)*2 +
                           src.at<_Tp>(i+1,j+1) - src.at<_Tp>(i+1,j-1);
            double gre_y=  src.at<_Tp>(i+1,j-1) - src.at<_Tp>(i-1,j-1) +
                           src.at<_Tp>(i+1,j)*2 - src.at<_Tp>(i-1,j)*2 +
                           src.at<_Tp>(i+1,j+1) - src.at<_Tp>(i-1,j+1);
            gre.at<float>(i-1,j-1)=sqrt(gre_x*gre_x+gre_y*gre_y);
            if(gre.at<float>(i-1,j-1)>max_gre)
                max_gre=gre.at<float>(i-1,j-1);
        }
    }
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            //double tempvalue = 1.0-(var.at<float>(i-1,j-1)/max_var+gre.at<float>(i-1,j-1)/max_gre)/2.0;
            //double tempvalue = 1.0-(var.at<float>(i-1,j-1)/max_var*gre.at<float>(i-1,j-1)/max_gre);
            double tempvalue = 1.0-var.at<float>(i-1,j-1)/max_var;
            dst.at<unsigned char>(i,j) = (unsigned char) (tempvalue*255.0);
        }
    }
    for(int i=1;i<src.rows-1;i++)
    {
        dst.at<unsigned char>(i,0)=dst.at<unsigned char>(i,1);
        dst.at<unsigned char>(i,src.cols-1)=dst.at<unsigned char>(i,src.cols-2);
    }
    for(int j=1;j<src.cols-1;j++)
    {
        dst.at<unsigned char>(0,j)=dst.at<unsigned char>(1,j);
        dst.at<unsigned char>(src.rows-1,j)=dst.at<unsigned char>(src.rows-2,j);
    }
    dst.at<unsigned char>(0,0)=dst.at<unsigned char>(1,1);
    dst.at<unsigned char>(src.rows-1,0)=dst.at<unsigned char>(src.rows-2,1);
    dst.at<unsigned char>(0,src.cols-1)=dst.at<unsigned char>(1,src.cols-2);
    dst.at<unsigned char>(src.rows-1,src.cols-1)=dst.at<unsigned char>(src.rows-2,src.cols-2);
    //cout<<max<<" "<<min<<std::endl;
}

//CS_LBP
template <typename _Tp>
void lbp::OCSLBP_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    int threshold=5;
    double diff=0;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            //_Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            diff=src.at<_Tp>(i-1,j-1)-src.at<_Tp>(i+1,j+1);
            code |= ( diff> threshold ) << 3;
            diff=src.at<_Tp>(i-1,j)-src.at<_Tp>(i+1,j);
            code |= ( diff> threshold ) << 2;
            diff=src.at<_Tp>(i-1,j+1)-src.at<_Tp>(i+1,j-1);
            code |= ( diff> threshold ) << 1;
            diff=src.at<_Tp>(i,j+1)-src.at<_Tp>(i,j-1) ;
            code |= ( diff> threshold ) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

//LTP
template <typename _Tp>
void lbp::OLTP_(const Mat& src, Mat& dst) {
    unsigned char lut[8][3];
    unsigned char cnt = 0;
    /*for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            lut[i][j] = cnt++;
            //cout<<(int)lut[i][j]<<" ";
        }
        cnt++; //we skip the 4th number (only three patterns)
    }*/
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 8; i++)
        {
            lut[i][j] = j*8+i;
            cout<<(int)lut[i][j]<<" ";
        }
        //cnt++; //we skip the 4th number (only three patterns)
    }
    double threshold=3.0;
    double thresholdneg=-1.0*threshold;
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    double diff;
    double max=0;
    double min=255;
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            diff=src.at<_Tp>(i-1,j-1) - center;
            //cout<<diff<<" ";
            if(diff>threshold)
                code=lut[0][0];
            else if(diff<thresholdneg)
                code=lut[0][2];
            else
                code=lut[0][1];
            diff=src.at<_Tp>(i-1,j) - center;
            if(diff>threshold)
                code+=lut[1][0];
            else if(diff<thresholdneg)
                code+=lut[1][2];
            else
                code+=lut[1][1];
            diff=src.at<_Tp>(i-1,j+1) - center;
            if(diff>threshold)
                code+=lut[2][0];
            else if(diff<thresholdneg)
                code+=lut[2][2];
            else
                code+=lut[2][1];
            diff=src.at<_Tp>(i,j+1) - center;
            if(diff>threshold)
                code+=lut[3][0];
            else if(diff<thresholdneg)
                code+=lut[3][2];
            else
                code+=lut[3][1];
            diff=src.at<_Tp>(i+1,j+1) - center;
            if(diff>threshold)
                code+=lut[4][0];
            else if(diff<thresholdneg)
                code+=lut[4][2];
            else
                code+=lut[4][1];
            diff=src.at<_Tp>(i+1,j) - center;
            if(diff>threshold)
                code+=lut[5][0];
            else if(diff<thresholdneg)
                code+=lut[5][2];
            else
                code+=lut[5][1];
            diff=src.at<_Tp>(i+1,j-1) - center;
            if(diff>threshold)
                code+=lut[6][0];
            else if(diff<thresholdneg)
                code+=lut[6][2];
            else
                code+=lut[6][1];
            diff=src.at<_Tp>(i,j-1) - center;
            if(diff>threshold)
                code+=lut[7][0];
            else if(diff<thresholdneg)
                code+=lut[7][2];
            else
                code+=lut[7][1];

            dst.at<unsigned char>(i-1,j-1) = code;
            //cout<<(int)code<<" ";
            if(code>max)
                max=(double)code;
            if(code<min)
                min=(double)code;
        }
    }
    cout<<max<<" "<<min<<std::endl;
}

template <typename _Tp>
void lbp::ELBP_(const Mat& src, Mat& dst, int radius, int neighbors) {
	neighbors = max(min(neighbors,31),1); // set bounds...
	// Note: alternatively you can switch to the new OpenCV Mat_
	// type system to define an unsigned int matrix... I am probably
	// mistaken here, but I didn't see an unsigned int representation
	// in OpenCV's classic typesystem...
	dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
	for(int n=0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
		float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 =      tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 =      tx  *      ty;
		// iterate through your data
		for(int i=radius; i < src.rows-radius;i++) {
			for(int j=radius;j < src.cols-radius;j++) {
				float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) && (abs(t-src.at<_Tp>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

template <typename _Tp>
void lbp::VARLBP_(const Mat& src, Mat& dst, int radius, int neighbors) {
	max(min(neighbors,31),1); // set bounds
	dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32FC1); //! result
	// allocate some memory for temporary on-line variance calculations
	Mat _mean = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat _delta = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat _m2 = Mat::zeros(src.rows, src.cols, CV_32FC1);
	for(int n=0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
		float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 =      tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 =      tx  *      ty;
		// iterate through your data
		for(int i=radius; i < src.rows-radius;i++) {
			for(int j=radius;j < src.cols-radius;j++) {
				float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
				_delta.at<float>(i,j) = t - _mean.at<float>(i,j);
				_mean.at<float>(i,j) = (_mean.at<float>(i,j) + (_delta.at<float>(i,j) / (1.0*(n+1)))); // i am a bit paranoid
				_m2.at<float>(i,j) = _m2.at<float>(i,j) + _delta.at<float>(i,j) * (t - _mean.at<float>(i,j));
			}
		}
	}
	// calculate result
	for(int i = radius; i < src.rows-radius; i++) {
		for(int j = radius; j < src.cols-radius; j++) {
			dst.at<float>(i-radius, j-radius) = _m2.at<float>(i,j) / (1.0*(neighbors-1));
		}
	}
}

// now the wrapper functions
void lbp::OLBP(const Mat& src, Mat& dst) {
	switch(src.type()) {
		case CV_8SC1: OLBP_<char>(src, dst); break;
		case CV_8UC1: OLBP_<unsigned char>(src, dst); break;
		case CV_16SC1: OLBP_<short>(src, dst); break;
		case CV_16UC1: OLBP_<unsigned short>(src, dst); break;
		case CV_32SC1: OLBP_<int>(src, dst); break;
		case CV_32FC1: OLBP_<float>(src, dst); break;
		case CV_64FC1: OLBP_<double>(src, dst); break;
	}
}

// CS_LBP
void lbp::OCSLBP(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OCSLBP_<char>(src, dst); break;
        case CV_8UC1: OCSLBP_<unsigned char>(src, dst); break;
        case CV_16SC1: OCSLBP_<short>(src, dst); break;
        case CV_16UC1: OCSLBP_<unsigned short>(src, dst); break;
        case CV_32SC1: OCSLBP_<int>(src, dst); break;
        case CV_32FC1: OCSLBP_<float>(src, dst); break;
        case CV_64FC1: OCSLBP_<double>(src, dst); break;
    }
}

//LTP
void lbp::OLTP(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OLTP_<char>(src, dst); break;
        case CV_8UC1: OLTP_<unsigned char>(src, dst); break;
        case CV_16SC1: OLTP_<short>(src, dst); break;
        case CV_16UC1: OLTP_<unsigned short>(src, dst); break;
        case CV_32SC1: OLTP_<int>(src, dst); break;
        case CV_32FC1: OLTP_<float>(src, dst); break;
        case CV_64FC1: OLTP_<double>(src, dst); break;
    }
}

//WLD
void lbp::OWLD(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OWLD_<char>(src, dst); break;
        case CV_8UC1: OWLD_<unsigned char>(src, dst); break;
        case CV_16SC1: OWLD_<short>(src, dst); break;
        case CV_16UC1: OWLD_<unsigned short>(src, dst); break;
        case CV_32SC1: OWLD_<int>(src, dst); break;
        case CV_32FC1: OWLD_<float>(src, dst); break;
        case CV_64FC1: OWLD_<double>(src, dst); break;
    }
}

//Local Homogeneity
void lbp::OHOMOGENEITY(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OHOMOGENEITY_<char>(src, dst); break;
        case CV_8UC1: OHOMOGENEITY_<unsigned char>(src, dst); break;
        case CV_16SC1: OHOMOGENEITY_<short>(src, dst); break;
        case CV_16UC1: OHOMOGENEITY_<unsigned short>(src, dst); break;
        case CV_32SC1: OHOMOGENEITY_<int>(src, dst); break;
        case CV_32FC1: OHOMOGENEITY_<float>(src, dst); break;
        case CV_64FC1: OHOMOGENEITY_<double>(src, dst); break;
    }
}

//Local Homogeneity of Texture
void lbp::OHOMOGENEITYOFTEXTURE(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OHOMOGENEITYOFTEXTURE_<char>(src, dst); break;
        case CV_8UC1: OHOMOGENEITYOFTEXTURE_<unsigned char>(src, dst); break;
        case CV_16SC1: OHOMOGENEITYOFTEXTURE_<short>(src, dst); break;
        case CV_16UC1: OHOMOGENEITYOFTEXTURE_<unsigned short>(src, dst); break;
        case CV_32SC1: OHOMOGENEITYOFTEXTURE_<int>(src, dst); break;
        case CV_32FC1: OHOMOGENEITYOFTEXTURE_<float>(src, dst); break;
        case CV_64FC1: OHOMOGENEITYOFTEXTURE_<double>(src, dst); break;
    }
}

//WLD_Short
void lbp::OWLDSHORT(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OWLDSHORT_<char>(src, dst); break;
        case CV_8UC1: OWLDSHORT_<unsigned char>(src, dst); break;
        case CV_16SC1: OWLDSHORT_<short>(src, dst); break;
        case CV_16UC1: OWLDSHORT_<unsigned short>(src, dst); break;
        case CV_32SC1: OWLDSHORT_<int>(src, dst); break;
        case CV_32FC1: OWLDSHORT_<float>(src, dst); break;
        case CV_64FC1: OWLDSHORT_<double>(src, dst); break;
    }
}

//WLD_Orientation
void lbp::OWLDORI(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OWLDORI_<char>(src, dst); break;
        case CV_8UC1: OWLDORI_<unsigned char>(src, dst); break;
        case CV_16SC1: OWLDORI_<short>(src, dst); break;
        case CV_16UC1: OWLDORI_<unsigned short>(src, dst); break;
        case CV_32SC1: OWLDORI_<int>(src, dst); break;
        case CV_32FC1: OWLDORI_<float>(src, dst); break;
        case CV_64FC1: OWLDORI_<double>(src, dst); break;
    }
}

void lbp::ELBP(const Mat& src, Mat& dst, int radius, int neighbors) {
	switch(src.type()) {
		case CV_8SC1: ELBP_<char>(src, dst, radius, neighbors); break;
		case CV_8UC1: ELBP_<unsigned char>(src, dst, radius, neighbors); break;
		case CV_16SC1: ELBP_<short>(src, dst, radius, neighbors); break;
		case CV_16UC1: ELBP_<unsigned short>(src, dst, radius, neighbors); break;
		case CV_32SC1: ELBP_<int>(src, dst, radius, neighbors); break;
		case CV_32FC1: ELBP_<float>(src, dst, radius, neighbors); break;
		case CV_64FC1: ELBP_<double>(src, dst, radius, neighbors); break;
	}
}

void lbp::VARLBP(const Mat& src, Mat& dst, int radius, int neighbors) {
	switch(src.type()) {
		case CV_8SC1: VARLBP_<char>(src, dst, radius, neighbors); break;
		case CV_8UC1: VARLBP_<unsigned char>(src, dst, radius, neighbors); break;
		case CV_16SC1: VARLBP_<short>(src, dst, radius, neighbors); break;
		case CV_16UC1: VARLBP_<unsigned short>(src, dst, radius, neighbors); break;
		case CV_32SC1: VARLBP_<int>(src, dst, radius, neighbors); break;
		case CV_32FC1: VARLBP_<float>(src, dst, radius, neighbors); break;
		case CV_64FC1: VARLBP_<double>(src, dst, radius, neighbors); break;
	}
}

// now the Mat return functions
Mat lbp::OLBP(const Mat& src) { Mat dst; OLBP(src, dst); return dst; }
Mat lbp::ELBP(const Mat& src, int radius, int neighbors) { Mat dst; ELBP(src, dst, radius, neighbors); return dst; }
Mat lbp::VARLBP(const Mat& src, int radius, int neighbors) { Mat dst; VARLBP(src, dst, radius, neighbors); return dst; }




