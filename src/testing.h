#ifndef CORNER_DETECTION_PROJECT_TESTING_H
#define CORNER_DETECTION_PROJECT_TESTING_H

#include "opencv2/opencv.hpp"
#include "Susan.h"
#include "ShiTomasi.h"
#include "image_manipulation.h"

int count_matches(const vector<Point2f>& custom_corners, const vector<Point2f>& opencv_corners, int epsilon);
#endif //CORNER_DETECTION_PROJECT_TESTING_H
