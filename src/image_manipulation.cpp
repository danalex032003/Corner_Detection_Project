#include "image_manipulation.h"

using namespace std;
using namespace cv;

Mat BGR2Gray(Mat source) {
    /*
     * Convert the image from BGR to grayscale
     */
    Mat gray = Mat::zeros(source.size(), CV_8UC1);
    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            gray.at<uchar>(i, j) = (source.at<Vec3b>(i, j)[0] +
                                  source.at<Vec3b>(i, j)[1] +
                                  source.at<Vec3b>(i, j)[2]) / 3;
        }
    }
    return gray;
}