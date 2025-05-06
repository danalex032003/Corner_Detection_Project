#include <opencv2/opencv.hpp>
#include "src/image_manipulation.h"
#include "src/ShiTomasi.h"

using namespace std;
using namespace cv;

int main() {

    //Mat image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\cubes_colored.jpg)", IMREAD_COLOR_BGR);
    //Mat image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\flat,750x,075,f-pad,750x1000,f8f8f8.jpg)", IMREAD_COLOR_BGR);
    //Mat image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\yellow-black-square-checkered-check-flag-pattern-grid-texture-checkerboard-vector.jpg)", IMREAD_COLOR_BGR);
    //Mat gray_image = BGR2Gray(image);
    //Mat gray_image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\Corner.png)", IMREAD_GRAYSCALE);
    //Mat gray_image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\star.bmp)", IMREAD_GRAYSCALE);
    //Mat gray_image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\Checkerboard_pattern.png)", IMREAD_GRAYSCALE);
    //Mat gray_image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\grayscale-photo-of-building.jpg)", IMREAD_GRAYSCALE);
    Mat gray_image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\grayscale-photo-of-high-rise-buildings.jpg)", IMREAD_GRAYSCALE);
    //Mat gray_image = imread(R"(D:\UTCN\An3\Sem2\PI\Corner_Detection_Project\images\checkers.png)", IMREAD_GRAYSCALE);
    gradient_mat gradient = compute_gradient(gray_image);
    vector<vector<Mat>> auto_correlation_mat = compute_auto_correlation(gradient);
    Mat corners_image = compute_corner_value(auto_correlation_mat, gray_image, 0.1);
    imshow("Corners Image", corners_image);
    imshow("Grayscale image", gray_image);
    waitKey();
}