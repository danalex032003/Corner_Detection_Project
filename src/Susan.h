#ifndef CORNER_DETECTION_PROJECT_SUSAN_H
#define CORNER_DETECTION_PROJECT_SUSAN_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const double SUSAN_GEOMETRIC_THRESHOLD_FACTOR = 0.75;


typedef struct {
    vector<vector<double>> mask;
    int pixel_no;
} susan_mask;

susan_mask get_susan_mask(int radius);

double compute_similarity_function(double center, double neighbor, double t);
double compute_usan(Mat source, int x, int y, int radius, double t, vector<vector<double>> mask);
double compute_corner_response(Mat source, int x, int y, int radius, double t, int pixel_no, vector<vector<double>> mask);
Mat compute_corner_response_map(Mat source, int radius, double t);
vector<Point2f> get_corners(Mat corner_response_map, double threshold);
double compute_local_stddev(const Mat& image, int x, int y, int radius, const vector<vector<double>>& mask);
vector<Point2f> run_susan(string filepath);

#endif //CORNER_DETECTION_PROJECT_SUSAN_H
