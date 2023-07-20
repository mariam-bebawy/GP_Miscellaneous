#include <iostream>
#include <fstream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include <omp.h>
#include <chrono>
#include <time.h>


using namespace std;
using namespace cv;
using namespace cv::cuda;

int main()
{

    Mat frame0;
    Mat frame1;
    Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
    //Capture camera
    VideoCapture cap("videos/pnas.2109838119.sm03.mp4");
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second camera : " << fps << endl;

    // Number of frames to capture
    int num_frames = 1;

    // Start and end times
    //clock_t startopticalflow;
    //clock_t endopticalflow;
    clock_t start, end;
    double ms, fpsLive;
    for (;;) {
        start = clock();
        //startopticalflow = clock();
        Mat image;
        cap >> image;

        cv::cvtColor(image, frame0, cv::COLOR_BGR2GRAY);

        if (frame1.empty()) {
            frame0.copyTo(frame1);
        }
        else {
            

            Mat flow;
            //Put Mat into GpuMat
            GpuMat GpuImg0(frame0);
            GpuMat GpuImg1(frame1);
            //Prepare space for output
            GpuMat gflow(frame0.size(), CV_32FC2);
            // chrono time to calculate the the needed time to compute and
            // draw the optical flow result
            // Calculate optical flow
            farn->calc(GpuImg0, GpuImg1, gflow);
            // GpuMat to Mat
            gflow.download(flow);
            //endopticalflow = clock();

            //for (int y = 0; y < image.rows - 1; y += 15){
            //    for (int x = 0; x < image.cols - 1; x += 15) {
            //        // get the flow from y, x position * 10 for better visibility
            //        const Point2f flowatxy = flow.at<Point2f>(y, x);
            //        // draw line at flow direction
            //        arrowedLine(image, Point(x, y), Point(cvRound(x + flowatxy.x),
            //            cvRound(y + flowatxy.y)), Scalar(0, 255, 0), 2);
            //        // draw initial point  https://funvision.blogspot.com
            //        circle(image, Point(x, y), 1, Scalar(0, 0, 255), -1);
            //    }
            //}

            //clock_t startvisualization;
            //clock_t endvisualization;
            //startvisualization = clock();

            Mat flow_parts[2];
            split(flow, flow_parts);

            // Convert the algorithm's output into Polar coordinates
            Mat magnitude, angle, magn_norm;
            cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));

            // Build hsv image
            Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);

            // Display the results
            cv::cvtColor(hsv8, bgr, COLOR_HSV2BGR);
            //endvisualization = clock();
            end = clock();


            //double secondsofopticalflow = (double(endopticalflow) - double(startopticalflow)) / double(CLOCKS_PER_SEC);
            //double secondsofvisualization = (double(endvisualization) - double(startvisualization)) / double(CLOCKS_PER_SEC);

            //cout << "Time taken for oprical flow : " << secondsofopticalflow << " seconds" << endl;
            //cout << "Time taken for visualization : " << secondsofvisualization << " seconds" << endl;
            double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
            fpsLive = double(num_frames) / double(seconds);
            //cout << "full time: " << seconds << " seconds" << endl;


            //cout << "Estimated frames per second : " << fpsLive << endl;
            putText(image, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2);
            imshow("flow", bgr);
            imshow("Display window", image);
            waitKey(1);
            // Save frame0 to frame1 to for next round
            // https://funvision.blogspot.com
            frame0.copyTo(frame1);
        }
    }
}