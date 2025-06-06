#include "ShiTomasi.h"
#include "image_manipulation.h"

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
                    x += value * (float) sobel_x.at<int>(dx + 1, dy + 1);
                    y += value * (float) sobel_y.at<int>(dx + 1, dy + 1);
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

vector<Point2f> selected;

Mat compute_corner_value(vector<vector<Mat>> auto_correlation_mat_per_pixel, Mat source, float threshold, int max_corners, float min_distance) {
    int rows = source.rows;
    int cols = source.cols;

    Mat corner_response = Mat::zeros(rows, cols, CV_32F);

    Mat final_image;
    cv::cvtColor(source, final_image, cv::COLOR_GRAY2BGR);

    float max_value = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Mat eigenvalues;
            eigen(auto_correlation_mat_per_pixel[i][j], eigenvalues);
            float min_lambda = min(eigenvalues.at<float>(0), eigenvalues.at<float>(1));
            corner_response.at<float>(i, j) = min_lambda;
            max_value = max(max_value, min_lambda);
        }
    }

    vector<Point2f> keypoints;
    int window_size = 3;
    int half_window = window_size / 2;

    for (int i = half_window; i < rows - half_window; ++i) {
        for (int j = half_window; j < cols - half_window; ++j) {
            float val = corner_response.at<float>(i, j);

            if (val < threshold * max_value)
                continue;

            bool is_local_max = true;
            for (int dx = -half_window; dx <= half_window; ++dx) {
                for (int dy = -half_window; dy <= half_window; ++dy) {
                    if (dx == 0 && dy == 0) continue;

                    if (corner_response.at<float>(i + dx, j + dy) >= val) {
                        is_local_max = false;
                        break;
                    }
                }
                if (!is_local_max) break;
            }

            if (is_local_max) {
                keypoints.emplace_back(j, i);
            }
        }
    }

    std::sort(keypoints.begin(), keypoints.end(), [&](const Point& a, const Point& b) {
        return corner_response.at<float>(a) > corner_response.at<float>(b);
    });


    for (const auto& pt : keypoints) {
        bool too_close = false;
        for (const auto& sel : selected) {
            if (norm(pt - sel) < min_distance) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            selected.push_back(pt);
            if ((int)selected.size() == max_corners) break;
        }
    }

    for (const auto& pt : selected) {
        circle(final_image, pt, 3, Scalar(0, 0, 255), -1);
    }

    return final_image;
}

vector<Point2f> run_shi_tomasi(string filepath) {
    Mat source = imread(filepath, IMREAD_COLOR);
    source = BGR2Gray(source);
    apply_gaussian_filtering_1D(source, 3);
    gradient_mat gradient = compute_gradient(source);
    vector<vector<Mat>> auto_correlation_mat = compute_auto_correlation(gradient);
    Mat corners_image = compute_corner_value(auto_correlation_mat, source, 0.01, 100, 10.0);
    imshow("Shi-Tomasi", corners_image);
    return selected;
}