#include <opencv2/imgproc.hpp>  // Includes OpenCV functions for image processing.
#include <opencv2/highgui.hpp>  // Includes high-level GUI functions to display images.
#include <opencv2/imgcodecs.hpp> // Includes image coding and decoding functions.
#include <iostream>             // Includes standard I/O stream objects.
#include <chrono>               // Includes time-related functions and classes for measuring time.
#include <omp.h>                // Includes functions from OpenMP for parallel programming.
#include <mpi.h>

using namespace cv;
using namespace std;

void accumulateVotes(const cv::Mat& img, int startRow, int endRow, int width, int rhoSize, int thetaSize, double maxDist, double rhoRes, double thetaRes, std::vector<int>& houghSpace) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            if (img.at<uchar>(y, x) > 0) {
                for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                    double theta = thetaIdx * thetaRes;
                    double rho = x * std::cos(theta) + y * std::sin(theta);
                    int rhoIdx = static_cast<int>(std::round((rho + maxDist) / rhoRes));
                    houghSpace[rhoIdx * thetaSize + thetaIdx]++;
                }
            }
        }
    }
}

std::vector<cv::Vec2f> parallel_hough_mpi(const cv::Mat& img, double rhoRes, double thetaRes, int threshold) {
    int width = img.cols;
    int height = img.rows;
    double maxDist = std::sqrt(width * width + height * height);
    int rhoSize = static_cast<int>(ceil(2 * maxDist / rhoRes));
    int thetaSize = static_cast<int>(ceil(CV_PI / thetaRes));

    int numProcs, procId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    int chunkSize = height / numProcs;
    int startRow = procId * chunkSize;
    int endRow = (procId == numProcs - 1)? height : startRow + chunkSize;

    std::vector<int> houghSpace(rhoSize * thetaSize);

    accumulateVotes(img, startRow, endRow, width, rhoSize, thetaSize, maxDist, rhoRes, thetaRes, houghSpace);

    std::vector<int> globalHoughSpace(rhoSize * thetaSize);
    MPI_Allreduce(houghSpace.data(), globalHoughSpace.data(), rhoSize * thetaSize, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<cv::Vec2f> lines;
    if (procId == 0) {
        for (int rhoIdx = 0; rhoIdx < rhoSize; ++rhoIdx) {
            for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                if (globalHoughSpace[rhoIdx * thetaSize + thetaIdx] > threshold) {
                    lines.push_back(cv::Vec2f((rhoIdx * rhoRes) - maxDist, thetaIdx * thetaRes));
                }
            }
        }
    }


    return lines;
}
