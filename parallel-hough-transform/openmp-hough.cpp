#include <opencv2/imgproc.hpp>  // Includes OpenCV functions for image processing.
#include <opencv2/highgui.hpp>  // Includes high-level GUI functions to display images.
#include <opencv2/imgcodecs.hpp> // Includes image coding and decoding functions.
#include <omp.h>                // Includes functions from OpenMP for parallel programming.
#include <mpi.h>

using namespace cv;
using namespace std;

vector<Vec2f> omp_hough(const Mat& img, double rhoRes, double thetaRes, int threshold, vector<Vec2f>* lines = nullptr) {

    int width = img.cols;  // Gets the number of columns in the image (image width).
    int height = img.rows; // Gets the number of rows in the image (image height).

    double maxDist = std::sqrt(width * width + height * height); // Computes the maximum distance from the origin to the image corner.

    int rhoSize = static_cast<int>(ceil(2 * maxDist / rhoRes)); // Calculate number of bins for rho.
    int thetaSize = static_cast<int>(ceil(CV_PI / thetaRes));   // Calculate number of bins for theta.
    Mat houghSpace = Mat::zeros(rhoSize, thetaSize, CV_32SC1); // Create a 2D array to accumulate votes in Hough space.

    #pragma omp parallel num_threads(8)
    {
        Mat localHoughSpace = Mat::zeros(rhoSize, thetaSize, CV_32SC1);
        
        #pragma omp single 
        {
            printf("Number of threads: %d\n", omp_get_num_threads());
        }

        #pragma omp for schedule(dynamic, 10) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (img.at<uchar>(y, x) > 0) { // Process only edge pixels (>0).
                    #pragma omp parallel for schedule(dynamic, 10)
                    for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                        double theta = thetaIdx * thetaRes;
                        double rho = x * std::cos(theta) + y * std::sin(theta);
                        int rhoIdx = static_cast<int>(std::round((rho + maxDist) / rhoRes));
                        localHoughSpace.at<int>(rhoIdx, thetaIdx)++; // Vote in local Hough space.
                    }
                }
            }
        }

        #pragma omp critical 
        {
            houghSpace += localHoughSpace;
        }
    }

    for (int rhoIdx = 0; rhoIdx < rhoSize; ++rhoIdx) {
        for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
            if (houghSpace.at<int>(rhoIdx, thetaIdx) > threshold) { // Check if a cell in Hough space exceeds the threshold.
                lines->push_back(Vec2f((rhoIdx * rhoRes) - maxDist, thetaIdx * thetaRes)); // Store the line parameters.
            }
        }
    }

    return *lines; 
}

