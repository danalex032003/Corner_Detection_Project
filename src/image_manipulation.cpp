#include "image_manipulation.h"

using namespace std;
using namespace cv;

Mat BGR2Gray(Mat source) {
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

vector<float> compute_kernel_1D(int kernel_size) {
    vector<float> kernel;

    float sigma = (float) kernel_size / 6.0f;
    float sum = 0.0f;
    int center = kernel_size / 2;
    for (int i = 0; i < kernel_size; ++i) {
        float value = exp(- (float) ((i - center) * (i - center))) / (2 * sigma * sigma);
        value /= (float) (sqrt(2 * CV_PI) * sigma);
        kernel.push_back(value);
        sum += value;
    }

    for (int i = 0; i < kernel_size; ++i) {
        kernel.at(i) /= sum;
    }

    return kernel;
}

Mat apply_gaussian_filtering_1D(Mat source, int kernel_size) {
    Mat result = source.clone();

    int border = kernel_size / 2;
    vector<float> kernel = compute_kernel_1D(kernel_size);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = border; j < source.cols - border; ++j) {

            float sum = 0.0f;
            for (int k = -border; k <= border; ++k) {
                sum += (float) source.at<uchar>(i, j + k) * kernel.at(k + border);
            }
            result.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }

    for (int i = border; i < source.rows - border; ++i) {
        for (int j = 0; j < source.cols; ++j) {

            float sum = 0.0f;
            for (int k = -border; k <= border; ++k) {
                sum += (float) source.at<uchar>(i + k, j) * kernel.at(k + border);
            }
            result.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }

    return result;

}