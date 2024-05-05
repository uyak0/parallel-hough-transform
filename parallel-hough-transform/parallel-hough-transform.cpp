#include <opencv2/imgproc.hpp>  // Includes OpenCV functions for image processing.
#include <opencv2/highgui.hpp>  // Includes high-level GUI functions to display images.
#include <opencv2/imgcodecs.hpp> // Includes image coding and decoding functions.
#include <iostream>             // Includes standard I/O stream objects.
#include <chrono>               // Includes time-related functions and classes for measuring time.
#include <omp.h>                // Includes functions from OpenMP for parallel programming.
#include <mpi.h>

using namespace cv;            // Uses the cv namespace from OpenCV to avoid prefixing cv::.
using namespace std;           // Uses the std namespace to avoid prefixing std::.

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

vector<Vec2f> omp_hough(const Mat& img, double rhoRes, double thetaRes, int threshold, vector<Vec2f>* lines = nullptr) {

    int width = img.cols;  // Gets the number of columns in the image (image width).
    int height = img.rows; // Gets the number of rows in the image (image height).

    double maxDist = std::sqrt(width * width + height * height); // Computes the maximum distance from the origin to the image corner.

    int rhoSize = static_cast<int>(ceil(2 * maxDist / rhoRes)); // Calculate number of bins for rho.
    int thetaSize = static_cast<int>(ceil(CV_PI / thetaRes));   // Calculate number of bins for theta.
    Mat houghSpace = Mat::zeros(rhoSize, thetaSize, CV_32SC1); // Create a 2D array to accumulate votes in Hough space.

    #pragma omp parallel num_threads(4) // Parallel region starts with 4 threads.
    {
        #pragma omp single
        {
            cout << "Number of threads: " << omp_get_num_threads() << endl; // Print the number of threads.
        }
        #pragma omp for // Indicates that the loop should be divided among the threads.
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (img.at<uchar>(y, x) > 0) { // Process only edge pixels (>0).
                    for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                        double theta = thetaIdx * thetaRes;
                        double rho = x * std::cos(theta) + y * std::sin(theta);
                        int rhoIdx = static_cast<int>(std::round((rho + maxDist) / rhoRes));
                        #pragma omp atomic
                        houghSpace.at<int>(rhoIdx, thetaIdx)++; // Vote in Hough space.
                    }
                }
            }
        }
    }

    for (int rhoIdx = 0; rhoIdx < rhoSize; ++rhoIdx) {
        for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
            if (houghSpace.at<int>(rhoIdx, thetaIdx) > threshold) { // Check if a cell in Hough space exceeds the threshold.
                lines->push_back(Vec2f((rhoIdx * rhoRes) - maxDist, thetaIdx * thetaRes)); // Store the line parameters.
            }
        }
    }

    return *lines; // Return the detected lines.
}

vector<Vec2f> hough(const Mat& img, double rhoRes, double thetaRes, int threshold, vector<Vec2f>* lines = nullptr) {

    int width = img.cols;  // Gets the number of columns in the image (image width).
    int height = img.rows; // Gets the number of rows in the image (image height).

    double maxDist = std::sqrt(width * width + height * height); // Computes the maximum distance from the origin to the image corner.

    int rhoSize = static_cast<int>(ceil(2 * maxDist / rhoRes)); // Calculate number of bins for rho.
    int thetaSize = static_cast<int>(ceil(CV_PI / thetaRes));   // Calculate number of bins for theta.
    Mat houghSpace = Mat::zeros(rhoSize, thetaSize, CV_32SC1); // Create a 2D array to accumulate votes in Hough space.

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (img.at<uchar>(y, x) > 0) { // Process only edge pixels (>0).
                for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                    double theta = thetaIdx * thetaRes;
                    double rho = x * std::cos(theta) + y * std::sin(theta);
                    int rhoIdx = static_cast<int>(std::round((rho + maxDist) / rhoRes));
                    #pragma omp atomic
                    houghSpace.at<int>(rhoIdx, thetaIdx)++; // Vote in Hough space.
                }
            }
        }
    }

    for (int rhoIdx = 0; rhoIdx < rhoSize; ++rhoIdx) {
        for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
            if (houghSpace.at<int>(rhoIdx, thetaIdx) > threshold) { // Check if a cell in Hough space exceeds the threshold.
                lines->push_back(Vec2f((rhoIdx * rhoRes) - maxDist, thetaIdx * thetaRes)); // Store the line parameters.
            }
        }
    }

    return *lines; // Return the detected lines.
}

int main(int argc, char** argv)
{
    Mat dst, cdst, cdstP; // Declare matrices for the destination image and color-converted images.
    const int threshold = 75;

    const char* default_file = "sudoku.png"; // Default file name.
    const char* filename = argc >= 2 ? argv[1] : default_file; // Determine filename from command line arguments.

    MPI_Init(&argc, &argv);

    Mat src = imread(filename , IMREAD_GRAYSCALE); // Load image in grayscale.

    if (src.empty()) { // Check if the image has been loaded successfully.
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1; // Exit if image is not loaded.
    }

    Canny(src, dst, 50, 200, 3); // Apply Canny edge detection algorithm.
    cvtColor(dst, cdst, COLOR_GRAY2BGR); // Convert grayscale image to color image.
    cdstP = cdst.clone(); // Clone the image for displaying results of a probabilistic Hough transform.


    //![hough_lines]
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection

    // Best threshold for hough-test.jpg is 120
    // Best threshold for sudoku.png is 170 
    // This for loop is for testing
    // for (int i = 0; i < 10; i++) {
    //     auto start = chrono::high_resolution_clock::now();
    //     mpi_hough(dst, 1, CV_PI / 180, threshold, &lines, 4);  // MPI parallel Hough Line Transform
    //     // omp_hough(dst, 1, CV_PI / 180, threshold, &lines);  // OpenMP parallel Hough Line Transform
    //     // hough(dst, 1, CV_PI / 180, threshold, &lines);    // Hough Line Transform
    //     // HoughLines(dst, lines, 1, CV_PI / 180, 150); // runs the actual detection
    //     auto end = chrono::high_resolution_clock::now();
    //     cout << "Time taken for hough transform: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // }
        auto start = chrono::high_resolution_clock::now();
        mpi_hough(dst, 1, CV_PI / 180, threshold, &lines);
        auto end = chrono::high_resolution_clock::now();
        cout << "Time taken for hough transform: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    //![imshow]
    // Show results
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA); // Draw detected lines.
    }

    imshow("Source", src); // Display original image.
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst); // Display lines.
    imwrite("standard_hough with threshold " + to_string(threshold) + ".jpg", cdst); // Save the image with detected lines.

    MPI_Finalize();
    return 0; // Return successful exit code.
}
