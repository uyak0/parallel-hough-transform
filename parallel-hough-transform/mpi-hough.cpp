#include <opencv2/imgproc.hpp>  // Includes OpenCV functions for image processing.
#include <opencv2/highgui.hpp>  // Includes high-level GUI functions to display images.
#include <opencv2/imgcodecs.hpp> // Includes image coding and decoding functions.
#include <iostream>             // Includes standard I/O stream objects.
#include <chrono>               // Includes time-related functions and classes for measuring time.
#include <omp.h>                // Includes functions from OpenMP for parallel programming.
#include <mpi.h>

using namespace cv;
using namespace std;

vector<Vec2f> mpi_hough(const Mat& img, double rhoRes, double thetaRes, int threshold, vector<Vec2f>* lines = nullptr) {
    int width = img.cols;
    int height = img.rows;

    double maxDist = std::sqrt(width * width + height * height);
    int rhoSize = static_cast<int>(ceil(2 * maxDist / rhoRes));
    int thetaSize = static_cast<int>(ceil(CV_PI / thetaRes));
    Mat houghSpace = Mat::zeros(rhoSize, thetaSize, CV_32SC1);

    int numProcs, procId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    int chunkSize = (height * width) / numProcs;
    int startRow = procId * chunkSize;
    int endRow = (procId == numProcs - 1)? height : startRow + chunkSize;

    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            if (img.at<uchar>(y, x) > 0) {
                for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                    double theta = thetaIdx * thetaRes;
                    double rho = x * std::cos(theta) + y * std::sin(theta);
                    int rhoIdx = static_cast<int>(std::round((rho + maxDist) / rhoRes));
                    int vote = 1;
                    MPI_Allreduce(&vote, &houghSpace.at<int>(rhoIdx, thetaIdx), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                }
            }
        }
    }


    for (int rhoIdx = 0; rhoIdx < rhoSize; ++rhoIdx) {
        for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
            if (houghSpace.at<int>(rhoIdx, thetaIdx) > threshold) {
                lines->push_back(Vec2f((rhoIdx * rhoRes) - maxDist, thetaIdx * thetaRes));
            }
        }
    }

    return *lines;
}
