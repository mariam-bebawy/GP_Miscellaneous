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
#include <omp.h>
#include <chrono>
#include <time.h>


using namespace std;
using namespace cv;
using namespace cv::cuda;

Mat convert_to_trajectories(vector<Mat> flowfields) {
    // flowfields is a Mat of shape rows*cols*time
    //initialize trajectories with a white image
    Mat initial_flow = flowfields[0];
    Mat trajectories(initial_flow.rows, initial_flow.cols, CV_8UC3, Scalar(255, 255, 255));
    //Vec3b color;
    //color = colorfields[0].at<Vec3b>(Point(0, 0));

    //loop over each pixel and follow its path

    for (int y = 0; y < initial_flow.rows; y += 3) {
        for (int x = 0; x < initial_flow.cols; x += 3) {
            Point prev = Point(x, y);
            Point current;

            for (int i = 1; i < flowfields.size(); i++) {
                Point2f displacement = flowfields[i].at<Point2f>(prev.y, prev.x);
                current = Point(cvRound(prev.x + displacement.x), cvRound(prev.y + displacement.y));
                //color = colorfields[i].at<Vec3b>(Point2f(x, y));
                line(trajectories, prev, current, Scalar(255 - 4 * i, 4 * i, 2 * i), 1);
                circle(trajectories, current, 0.5, Scalar(255 - 4 * i, 4 * i, 2 * i), 1);
                prev = current;

            }
        }
    }


    return trajectories;
}
int main(int argc, const char** argv)
{

    Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();

    // add your file name
    VideoCapture cap("videos/pnas.2109838119.sm03.mp4");
    double frame_Number = cap.get(CAP_PROP_FRAME_COUNT);
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second : " << fps << endl;
    cout << "total frames Frames  : " << frame_Number << endl;


    // Number of frames to capture
    int num_frames = 1;

    // Start and end times
    clock_t start;
    clock_t end;
    double ms, fpsLive;
    Mat flow, frame;
    // some faster than mat image container
    UMat  flowUmat, prevgray;
    vector<Mat>flow_fields;
    vector<Mat>color_fields;
    for (int k = 1; k < 80; k++)
    {
        start = clock();

        bool Is = cap.grab();
        if (Is == false) {
            // if video capture failed
            cout << "Video Capture Fail" << endl;
            break;
        }
        else {
            Mat img;
            Mat original;
            // capture frame from video file
            cap.retrieve(img, CAP_OPENNI_BGR_IMAGE);
            //rsize(img, img, Size(640, 480));
            // save original for later
            img.copyTo(original);
            // just make current frame gray
            cv::cvtColor(img, img, COLOR_BGR2GRAY);
            // For all optical flow you need a sequence of images.. Or at least 2 of them. Previous
                                    //and current frame
            GpuMat GpuImg0(img);
            GpuMat GpuImg1(prevgray);
            //Prepare space for output
            GpuMat gflow(img.size(), CV_32FC2);


            if (prevgray.empty() == false) {
                // Calculate optical flow
                farn->calc(GpuImg1, GpuImg0, gflow);
                // GpuMat to Mat
                gflow.download(flow);

                flow_fields.push_back(flow);

                Mat colors;

                Mat flow_parts[2];
                split(flow, flow_parts);

                // Convert the algorithm's output into Polar coordinates
                Mat magnitude, angle;
                cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
                angle *= ((1.f / 360.f) * (180.f / 255.f));
                Mat _hsv[3], hsv, hsv8, bgr;
                _hsv[0] = angle;
                _hsv[1] = Mat::ones(angle.size(), CV_32F);
                _hsv[2] = Mat::ones(angle.size(), CV_32F);;
                merge(_hsv, 3, hsv);
                hsv.convertTo(hsv8, CV_8U, 255.0);

                // Display the results
                cv::cvtColor(hsv8, colors, COLOR_HSV2BGR);
                //color_fields.push_back(colors);

                Vec3b color;
                color = colors.at<Vec3b>(Point(250, 300));

                cout << (float)color[0] << endl;
                Mat whiteimage(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

                for (int y = 0; y < original.rows; y += 5) {
                    for (int x = 0; x < original.cols; x += 5) {
                        if (original.at<Vec3b>(Point2f(x, y)) != Vec3b{ 0,0,0 }) {
                            const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
                            // draw line at flow direction
                            color = colors.at<Vec3b>(Point2f(x, y));
                            line(whiteimage, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar((float)color[0], (float)color[1], (float)color[2]));
                            // draw initial point
                            circle(whiteimage, Point(x, y), 1, Scalar((float)color[0], (float)color[1], (float)color[2]), -1);
                        }
                    }
                }
                end = clock();
                double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
                cout << "Time taken : " << seconds << " seconds" << endl;
                fpsLive = double(num_frames) / double(seconds);
                //fpsLive = double(num_frames) / double(ms*0.001);
                cout << "Estimated frames per second : " << fpsLive << endl;
                putText(whiteimage, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2);
                // draw the results
                namedWindow("prew", WINDOW_AUTOSIZE);
                imshow("prew", original);
                namedWindow("flow", WINDOW_AUTOSIZE);

                imshow("flow", whiteimage);



                // fill previous image again
                img.copyTo(prevgray);
            }
            else {
                // fill previous image in case prevgray.empty() == true
                img.copyTo(prevgray);
            }
            int key1 = waitKey(1);
        }
    }
    Mat trajectories = convert_to_trajectories((flow_fields));
    namedWindow("trajectories", WINDOW_AUTOSIZE);
    imshow("trajectories", trajectories);
    waitKey(0);

}