#ifndef CORNER_DETECTION_PROJECT_SHITOMASI_H
#define CORNER_DETECTION_PROJECT_SHITOMASI_H

#include <opencv2/opencv.hpp>
#include "testing.h"

using namespace std;
using namespace cv;

typedef struct {
    Mat gradient_mat_x;
    Mat gradient_mat_y;
}gradient_mat;

const Mat sobel_x = (Mat_<int>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

const Mat sobel_y = (Mat_<int>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1);

gradient_mat compute_gradient(Mat source);

vector<vector<Mat>> compute_auto_correlation(gradient_mat);

Mat compute_corner_value(vector<vector<Mat>> auto_correlation_mat_per_pixel, Mat source, float threshold, int max_corners, float min_distance);

vector<Point2f> run_shi_tomasi(string filepath);

#endif //CORNER_DETECTION_PROJECT_SHITOMASI_H
