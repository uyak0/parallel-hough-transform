#include <map>
#include <opencv2/imgproc.hpp>  // Includes OpenCV functions for image processing.
#include <opencv2/highgui.hpp>  // Includes high-level GUI functions to display images.
#include <opencv2/imgcodecs.hpp> // Includes image coding and decoding functions.
#include <iostream>             
#include <chrono>               // Includes time-related functions and classes for measuring time.
#include <omp.h>               
#include <mpi.h>
#include <string>

#include "openmp-hough.cpp"
#include "mpi-hough.cpp"
#include "serial-hough.cpp"

using namespace cv;           
using namespace std;         

enum ParallelizationMethod {
    openmp,
    mpi,
    serial 
};

map<string, ParallelizationMethod> parallelization_methods = {
    {"openmp", openmp},
    {"mpi", mpi},
    {"serial", serial}
};

int main(int argc, char** argv)
{
    Mat dst, cdst; // Declare matrices for the destination image and color-converted images.
    
    // Best threshold for hough-test.jpg is 120
    // Best threshold for sudoku.png is 170 
    int threshold = 120;

    const char* default_file = "sudoku.png"; // Default file name.
    const char* filename = argc >= 2 ? argv[1] : default_file; // Determine filename from command line arguments.
    const char* parallel_platform = argc >= 3 ? argv[2] : "openmp"; // Determine parallelization method from command line arguments.
    cout << "Parallelization method: " << parallel_platform << endl;

    Mat src = imread(filename , IMREAD_GRAYSCALE); // Load image in grayscale.
    ParallelizationMethod method = parallelization_methods[parallel_platform];
    cout << "Parallelization method: " << method << endl;

    if (src.empty()) { // Check if the image has been loaded successfully.
        printf(" Error opening image\n");
        printf(" Program Arguments: <image-path> <parallel-method>"); 
        return -1; // Exit if image is not loaded.
    }

    Canny(src, dst, 50, 200, 3); // Apply Canny edge detection algorithm, dst is the output image.
    cvtColor(dst, cdst, COLOR_GRAY2BGR); // Convert grayscale image to color image, cdst is the output image.

    //![hough_lines]
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection

    // This for loop is for testing
    MPI_Init(&argc, &argv);
    for (int i = 0; i < 10; i++) {

        switch (method) {
            case openmp: {
                cout << "Using OpenMP" << endl;

                auto start = chrono::high_resolution_clock::now();
                omp_hough(dst, 1, CV_PI / 180, threshold, &lines);  // OpenMP parallel Hough Line Transform
                auto end = chrono::high_resolution_clock::now();
                cout << "Time taken for openMP: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

                break;
            }

            case mpi: {
                cout << "Using MPI" << endl;

                auto start3 = chrono::high_resolution_clock::now();
                lines = parallel_hough_mpi(dst, 1, CV_PI / 180, threshold);
                auto end3 = chrono::high_resolution_clock::now();
                cout << "Time taken for MPI: " << chrono::duration_cast<chrono::milliseconds>(end3 - start3).count() << "ms" << endl;

                break;
            }

            case serial: {
                cout << "Using Serial" << endl;

                auto start2 = chrono::high_resolution_clock::now();
                hough(dst, 1, CV_PI / 180, threshold, &lines);    // Hough Line Transform
                auto end2 = chrono::high_resolution_clock::now();
                cout << "Time taken for Serial: " << chrono::duration_cast<chrono::milliseconds>(end2 - start2).count() << "ms" << endl;

                break;
            }

            default:
                cout << "Invalid parallelization method" << endl;
                return -1;
        }
    }
    MPI_Finalize();

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
    imwrite("images/" + string (filename) + "-with-threshold-" + to_string(threshold) + ".jpg", cdst); // Save the image with detected lines.

    return 0; 
}
