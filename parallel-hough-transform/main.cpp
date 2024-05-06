#include <opencv2/imgproc.hpp>  // Includes OpenCV functions for image processing.
#include <opencv2/highgui.hpp>  // Includes high-level GUI functions to display images.
#include <opencv2/imgcodecs.hpp> // Includes image coding and decoding functions.
#include <iostream>             
#include <chrono>               // Includes time-related functions and classes for measuring time.
#include <omp.h>               
#include <mpi.h>

#include "openmp-hough.cpp"
#include "mpi-hough.cpp"
#include "serial-hough.cpp"

using namespace cv;           
using namespace std;         

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
    for (int i = 0; i < 10; i++) {
        auto start = chrono::high_resolution_clock::now();
        // omp_hough(dst, 1, CV_PI / 180, threshold, &lines);  // OpenMP parallel Hough Line Transform
        hough(dst, 1, CV_PI / 180, threshold, &lines);    // Hough Line Transform
        // HoughLines(dst, lines, 1, CV_PI / 180, 150); // runs the actual detection
        auto end = chrono::high_resolution_clock::now();
        cout << "Time taken for hough transform: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
        // auto start = chrono::high_resolution_clock::now();
        // mpi_hough(dst, 1, CV_PI / 180, threshold, &lines);
        // auto end = chrono::high_resolution_clock::now();
        // cout << "Time taken for hough transform: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

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
    return 0; 
}
