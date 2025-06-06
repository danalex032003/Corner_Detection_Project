#include "Susan.h"
#include "image_manipulation.h"

double NMS_RESPONSE_THRESHOLD_FACTOR = 0.8;

susan_mask get_susan_mask(int radius) {
    const int size = radius * 2 + 1;
    vector<vector<double>> mask(size, vector<double>(size, 0));
    int pixel_no = 0;
    const int center = radius;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double dist = sqrt((j - center) * (j - center) + (i - center) * (i - center));
            if (dist <= radius) {
                mask[i][j] = 1;
                pixel_no++;
            }
        }
    }

    return { mask, pixel_no };
}

double compute_similarity_function(double center, double neighbor, double t) {
    return exp(-pow((neighbor - center) / t, 4));
}

bool is_in_image(int x, int cols, int y, int rows) {
    return x >= 0 && x < cols && y >= 0 && y < rows;
}


double compute_usan(Mat source, int x, int y, int radius, double t, vector<vector<double>> mask) {

    double n = 0.0;
    double center = source.at<uchar>(y, x);
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (mask[dy + radius][dx + radius]) {
                int nx = x + dx;
                int ny = y + dy;

                if (is_in_image(nx, source.cols, ny, source.rows)) {
                    double neighbor = source.at<uchar>(ny, nx);
                    n += compute_similarity_function(center, neighbor, t);
                }
            }
        }
    }

    return n;
}

double compute_corner_response(Mat source, int x, int y, int radius, double k, int pixel_no, vector<vector<double>> mask) {

    double usan_threshold = SUSAN_GEOMETRIC_THRESHOLD_FACTOR * pixel_no;
    double local_t = k * compute_local_stddev(source, x, y, radius, mask);
    local_t = clamp(local_t, 5.0, 60.0);
    double usan_area = compute_usan(source, x, y, radius, local_t, mask);

    return max(0.0, usan_threshold - usan_area);
}

Mat compute_corner_response_map(Mat source, int radius, double k) {

    Mat corner_response_map = Mat::zeros(source.size(), CV_32F);
    susan_mask mask = get_susan_mask(radius);
    for (int i = radius; i < source.rows - radius; ++i) {
        for (int j = radius; j < source.cols - radius; ++j) {
            corner_response_map.at<float>(i, j) = (float) compute_corner_response(
                    source, j, i, radius, k, mask.pixel_no, mask.mask);
        }
    }
    return corner_response_map;
}

bool is_local_maximum(Mat& map, int x, int y, int radius) {
    float val = map.at<float>(y, x);
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < map.cols && ny >= 0 && ny < map.rows) {
                if (map.at<float>(ny, nx) > val) {
                    return false;
                }
            }
        }
    }
    return true;
}

vector<Point2f> get_corners(Mat corner_response_map, double threshold) {

    vector<Point2f> corners;
    for (int i = 1; i < corner_response_map.rows - 1; ++i) {
        for (int j = 1; j < corner_response_map.cols - 1; ++j) {
            float pixel = corner_response_map.at<float>(i, j);

            if (pixel > threshold && is_local_maximum(corner_response_map, j, i, 1)) {
                corners.emplace_back(j, i);
            }
        }
    }
    return corners;
}

double compute_local_stddev(const Mat& image, int x, int y, int radius, const vector<vector<double>>& mask) {
    double mean = 0.0, sum_sq = 0.0;
    int count = 0;
    int center = radius;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (mask[dy + center][dx + center]) {
                int nx = x + dx;
                int ny = y + dy;
                if (is_in_image(nx, image.cols, ny, image.rows)) {
                    double val = image.at<uchar>(ny, nx);
                    mean += val;
                    count++;
                }
            }
        }
    }

    if (count == 0) return 1.0;

    mean /= count;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (mask[dy + center][dx + center]) {
                int nx = x + dx;
                int ny = y + dy;
                if (is_in_image(nx, image.cols, ny, image.rows)) {
                    double val = image.at<uchar>(ny, nx);
                    sum_sq += (val - mean) * (val - mean);
                }
            }
        }
    }

    return sqrt(sum_sq / count);
}

Mat anisotropic_diffusion(const Mat& src, int iterations, float lambda = 0.25f, float k = 15.0f) {
    CV_Assert(src.type() == CV_8UC1);

    Mat current;
    src.convertTo(current, CV_32F);

    for (int iter = 0; iter < iterations; ++iter) {
        Mat north = Mat::zeros(current.size(), CV_32F);
        Mat south = Mat::zeros(current.size(), CV_32F);
        Mat east = Mat::zeros(current.size(), CV_32F);
        Mat west = Mat::zeros(current.size(), CV_32F);

        north.rowRange(0, current.rows - 1) = current.rowRange(1, current.rows) - current.rowRange(0, current.rows - 1);
        south.rowRange(1, current.rows)     = current.rowRange(0, current.rows - 1) - current.rowRange(1, current.rows);
        east.colRange(1, current.cols)      = current.colRange(0, current.cols - 1) - current.colRange(1, current.cols);
        west.colRange(0, current.cols - 1)  = current.colRange(1, current.cols) - current.colRange(0, current.cols - 1);

        Mat n2 = north.mul(north), s2 = south.mul(south), e2 = east.mul(east), w2 = west.mul(west);

        Mat cN, cS, cE, cW;
        cv::exp(-n2 / (k * k), cN);
        cv::exp(-s2 / (k * k), cS);
        cv::exp(-e2 / (k * k), cE);
        cv::exp(-w2 / (k * k), cW);

        current += lambda * (cN.mul(north) + cS.mul(south) + cE.mul(east) + cW.mul(west));
    }

    Mat result;
    current.convertTo(result, CV_8U);
    return result;
}

vector<Point2f> filter_corners_with_distance(Mat& responseMap, vector<Point2f>& candidates, int maxCorners, double minDistance) {
    sort(candidates.begin(), candidates.end(), [&responseMap](const Point2f& a, const Point2f& b) {
        return responseMap.at<float>(a) > responseMap.at<float>(b);
    });

    vector<Point2f> filtered;
    for (const auto& pt : candidates) {
        bool keep = true;
        for (const auto& sel : filtered) {
            if (norm(pt - sel) < minDistance) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered.push_back(pt);
            if ((int)filtered.size() >= maxCorners)
                break;
        }
    }
    return filtered;
}

vector<Point2f> run_susan(string filepath) {

    Mat source = imread(filepath, IMREAD_COLOR);
    source = BGR2Gray(source);
    apply_gaussian_filtering_1D(source, 3);
    Mat mat = anisotropic_diffusion(source, 10, 0.25f, 20.0f);
    Mat result;
    int radius = 3;
    double k = 0.8;
    Mat corner_response_map = compute_corner_response_map(mat, radius, k);
    double max_response;
    minMaxLoc(corner_response_map, nullptr, &max_response);
    double threshold = NMS_RESPONSE_THRESHOLD_FACTOR * max_response;
    int maxCorners = 100;
    double minDistance = 10.0;

    vector<Point2f> allCorners = get_corners(corner_response_map, threshold);
    vector<Point2f> corners = filter_corners_with_distance(corner_response_map, allCorners, maxCorners, minDistance);
    cvtColor(mat, result, COLOR_GRAY2BGR);
    for (const auto& p : corners) {
        circle(result, p, 3, Scalar(0, 0, 255), -1);
    }
    imshow("SUSAN", result);
    return corners;
}

