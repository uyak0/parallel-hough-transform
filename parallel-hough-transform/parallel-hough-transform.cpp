#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace std;

std::vector<cv::Vec2f> houghTransform(const cv::Mat& img, double rhoRes, double thetaRes, int threshold) {
    // Get image dimensions
    int width = img.cols;
    int height = img.rows;

    // Calculate the maximum possible distance (diagonal length)
    double maxDist = std::sqrt(width * width + height * height);

    // Create Hough space
    int rhoSize = static_cast<int>(std::ceil(2 * maxDist / rhoRes));
    int thetaSize = static_cast<int>(std::ceil(CV_PI / thetaRes));
    cv::Mat houghSpace = cv::Mat::zeros(rhoSize, thetaSize, CV_32SC1);

    // Perform Hough Transform
    cout << omp_get_num_threads();
        #pragma	omp parallel for 
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Only consider edge pixels
                if (img.at<uchar>(y, x) > 0) {
                    for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
                        double theta = thetaIdx * thetaRes;
                        double rho = x * std::cos(theta) + y * std::sin(theta);
                        int rhoIdx = static_cast<int>(std::round((rho + maxDist) / rhoRes));
                        #pragma omp atomic
                        houghSpace.at<int>(rhoIdx, thetaIdx)++;
                    }
                }
            }
        }
 


    // Extract lines from Hough space
    std::vector<cv::Vec2f> lines;
    for (int rhoIdx = 0; rhoIdx < rhoSize; ++rhoIdx) {
        for (int thetaIdx = 0; thetaIdx < thetaSize; ++thetaIdx) {
            if (houghSpace.at<int>(rhoIdx, thetaIdx) > threshold) {
                lines.push_back(cv::Vec2f((rhoIdx * rhoRes) - maxDist, thetaIdx * thetaRes));
            }
        }
    }

    return lines;
}

int main(int argc, char** argv)
{
    // Declare the output variables
    Mat dst, cdst, cdstP;

    //![load]
    const char* default_file = "sudoku.png";
    const char* filename = argc >= 2 ? argv[1] : default_file;

    // Loads an image
    Mat src = imread("./sudoku.png", IMREAD_GRAYSCALE);

    // Check if image is loaded fine
    if (src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }
    //![load]

    //![edge_detection]
    // Edge detection
    Canny(src, dst, 50, 200, 3);
    //![edge_detection]

    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    //![hough_lines]
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
	auto start = chrono::high_resolution_clock::now();
	lines = houghTransform(dst, 1, CV_PI / 180, 150); // runs the actual detection
	//HoughLines(dst, lines, 1, CV_PI / 180, 150); // runs the actual detection
	auto end = chrono::high_resolution_clock::now();

	cout << "Standard Hough Line Transform: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    //![hough_lines]
    //![draw_lines]
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    //![draw_lines]

    ////![hough_lines_p]
    //// Probabilistic Line Transform
    //vector<Vec4i> linesP; // will hold the results of the detection
    //HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
    ////![hough_lines_p]
    ////![draw_lines_p]
    //// Draw the lines
    //for (size_t i = 0; i < linesP.size(); i++)
    //{
    //    Vec4i l = linesP[i];
    //    line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    //}
    ////![draw_lines_p]

    //![imshow]
    // Show results
    imshow("Source", src);
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    //imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    //![imshow]

    //![exit]
    // Wait and Exit
    waitKey();
    return 0;
    //![exit]
}
