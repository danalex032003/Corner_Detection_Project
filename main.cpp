#include <opencv2/opencv.hpp>
#include "src/image_manipulation.h"
#include "src/ShiTomasi.h"
#include "src/Susan.h"
#include "src/testing.h"

using namespace std;
using namespace cv;

vector<Point2f> run_shi_tomasi_built_in(string filepath) {
    Mat image = imread(filepath, IMREAD_COLOR);
    image = BGR2Gray(image);

    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 10;

    vector<Point2f> corners;
    goodFeaturesToTrack(
            image,
            corners,
            maxCorners,
            qualityLevel,
            minDistance
    );

    Mat result;
    cvtColor(image, result, COLOR_GRAY2BGR);
    for (const auto& pt : corners) {
        circle(result, pt, 3, Scalar(0, 0, 255), -1);
    }

    imshow("Shi-Tomasi Corners built in", result);

    return corners;
}

int main() {
    string filepath = "D:\\UTCN\\An3\\Sem2\\PI\\Corner_Detection_Project\\images\\left.jpg";

    vector<Point2f> shi_tomasi_corners = run_shi_tomasi(filepath);
    vector<Point2f> susan_corners = run_susan(filepath);

    vector<Point2f> opencv_corners = run_shi_tomasi_built_in(filepath);
    int matches_shi_tomasi = count_matches(shi_tomasi_corners, opencv_corners, 10);
    int matches_susan = count_matches(susan_corners, opencv_corners, 5);
    printf("Matches Shi-Tomasi: %d/100, Matches Susan: %d/100", matches_shi_tomasi, matches_susan);
    waitKey();
}