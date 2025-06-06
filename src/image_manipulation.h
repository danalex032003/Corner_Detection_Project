#ifndef CORNER_DETECTION_PROJECT_IMAGE_MANIPULATION_H
#define CORNER_DETECTION_PROJECT_IMAGE_MANIPULATION_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat BGR2Gray(Mat source);
vector<float> compute_kernel_1D(int kernel_size);
Mat apply_gaussian_filtering_1D(Mat source, int kernel_size);

#endif //CORNER_DETECTION_PROJECT_IMAGE_MANIPULATION_H
