#include "testing.h"

int count_matches(const vector<Point2f>& custom_corners, const vector<Point2f>& opencv_corners, int epsilon) {
    int matches = 0;
    for (const auto& susan_point : custom_corners) {
        for (const auto& opencv_point : opencv_corners) {
            if (norm(susan_point - opencv_point) <= epsilon) {
                matches++;
            }
        }
    }
    return matches;
}
