#include "ShiTomasi.h"

using namespace std;
using namespace cv;

gradient_mat compute_gradient(Mat source) {

    Mat gradient_mat_x = Mat::zeros(source.size(), CV_32F);
    Mat gradient_mat_y = Mat::zeros(source.size(), CV_32F);

    for (int i = 0; i < source.rows; ++i) {
        gradient_mat_x.at<float>(i, 0) = 0;
        gradient_mat_x.at<float>(i, source.cols - 1) = 0;
        gradient_mat_y.at<float>(i, 0) = 0;
        gradient_mat_y.at<float>(i, source.cols - 1) = 0;
    }

    for (int i = 0; i < source.cols; ++i) {
        gradient_mat_x.at<float>(0, i) = 0;
        gradient_mat_x.at<float>(source.rows - 1, i) = 0;
        gradient_mat_y.at<float>(0, i) = 0;
        gradient_mat_y.at<float>(source.rows - 1, i) = 0;
    }

    for (int i = 1; i < source.rows - 1; ++i) {
        for (int j = 1; j < source.cols - 1; ++j) {
            float x = 0, y = 0;
            for (int dx = -1; dx <= 1; ++dx) {

                for (int dy = -1; dy <= 1; ++dy) {
                    float value = source.at<uchar>(i + dx, j + dy);
                    x += value * sobel_x.at<int>(dx + 1, dy + 1);
                    y += value * sobel_y.at<int>(dx + 1, dy + 1);
                }
            }
            gradient_mat_x.at<float>(i, j) = x;
            gradient_mat_y.at<float>(i, j) = y;
        }
    }

    return {gradient_mat_x, gradient_mat_y};
}

vector<vector<Mat>> compute_auto_correlation(gradient_mat gradient) {
    int rows = gradient.gradient_mat_x.rows;
    int cols = gradient.gradient_mat_y.cols;
    vector<vector<Mat>> auto_correlation_mat_per_pixel(rows, vector<Mat>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            Mat auto_correlation_mat = Mat::zeros(2, 2, CV_32F);
            float gx = gradient.gradient_mat_x.at<float>(i, j);
            float gy = gradient.gradient_mat_y.at<float>(i, j);

            auto_correlation_mat.at<float>(0, 0) = gx * gx;
            auto_correlation_mat.at<float>(0, 1) = gx * gy;
            auto_correlation_mat.at<float>(1, 0) = gx * gy;
            auto_correlation_mat.at<float>(1, 1) = gy * gy;

            auto_correlation_mat_per_pixel[i][j] = auto_correlation_mat;
        }
    }
    return auto_correlation_mat_per_pixel;
}

Mat compute_corner_value(vector<vector<Mat>> auto_correlation_mat_per_pixel, Mat source, float threshold) {
    int rows = source.rows;
    int cols = source.cols;

    Mat corner_value_mat = Mat::zeros(rows, cols, CV_32F);

    Mat final_image;
    cv::cvtColor(source, final_image, cv::COLOR_GRAY2BGR);

    float max_value = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Mat eigenvalues;
            eigen(auto_correlation_mat_per_pixel[i][j], eigenvalues);
            float min_lambda = min(eigenvalues.at<float>(0), eigenvalues.at<float>(1));
            max_value = max(max_value, min_lambda);
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Mat eigenvalues;
            eigen(auto_correlation_mat_per_pixel[i][j], eigenvalues);
            float min_lambda = min(eigenvalues.at<float>(0), eigenvalues.at<float>(1));
            if (min_lambda > max_value * threshold) {
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        final_image.at<Vec3b>(i + ii, j + jj) = Vec3b(0, 0, 255);
                    }
                }
            }
        }
    }

    return final_image;
}