#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <time.h>
//#include"Conversion_from_flowfields _to _trajectories.cpp"

using namespace cv;
using namespace std;

Point ceildisplacement(Point2f disp) {
    Point ceild;
    if (disp.x <= 0) {
        ceild.x =  cvFloor( disp.x);

    }
    else {
        ceild.x = cvCeil( disp.x);

    }
    if (disp.y <= 0) {
        ceild.y = cvFloor(disp.y);

    }
    else {
        ceild.y = cvCeil(disp.y);

    }
    return ceild;

}

Mat convert_to_trajectories(vector<Mat> flowfields,Mat original_image) {
    // flowfields is a Mat of shape rows*cols*time
    //initialize trajectories with a white image
    Mat initial_flow = flowfields[0];
    Mat trajectories(initial_flow.rows, initial_flow.cols, CV_8UC3, Scalar(255, 255, 255));                
    initial_flow = flowfields[0];
    VideoCapture cap("videos/pnas.2109838119.sm03.mp4");
    Mat skipped;
    for (int r = 0; r < 300; r++) {
        bool Is = cap.grab();
        cap.retrieve(skipped, CAP_OPENNI_BGR_IMAGE);

    }
    Mat initial, original;
    cap.retrieve(original, CAP_OPENNI_BGR_IMAGE);
    //rsize(img, img, Size(640, 480));
    // save original for later
    original.copyTo(initial);


    

    //loop over each pixel and follow its path
   // cout << "flowfields_num" << flowfields.size() << endl;
    for (int y = 0; y < initial_flow.rows; y +=50 ) {
        for (int x = 0; x < initial_flow.cols; x += 100) {
            Point prev = Point(x, y);
            Point current;
            
            for (int i = 0; i < flowfields.size(); i++) {
                if (original.at<Vec3b>(prev) != Vec3b{ 0,0,0 }) {

                    Point displacement = ceildisplacement(flowfields[i].at<Point2f>(prev.y, prev.x))*10;
                    current = Point((x + displacement.x), (y + displacement.y));
                    arrowedLine(trajectories, prev, current, Scalar(255 -  i,  i,   i));

                }
                prev = current;
                cap.retrieve(original, CAP_OPENNI_BGR_IMAGE);

                
                //cout << "flow num " << i << endl;
            }
            cout << x << endl;
            VideoCapture cap("videos/pnas.2109838119.sm03.mp4");
            Mat skipped;
            for (int r = 0; r < 300; r++) {
                bool Is = cap.grab();
                cap.retrieve(skipped, CAP_OPENNI_BGR_IMAGE);

            }
            Mat initial, original;
            cap.retrieve(original, CAP_OPENNI_BGR_IMAGE);
            //rsize(img, img, Size(640, 480));
            // save original for later
            original.copyTo(initial);
        }
        cout << y << endl;

    }


    return trajectories;
}
int main(int argc, const char** argv)
{   


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
    Mat skipped;

    for (int r = 0; r < 300; r++) {
        bool Is = cap.grab();
        cap.retrieve(skipped, CAP_OPENNI_BGR_IMAGE);

    }

    namedWindow("skip", WINDOW_AUTOSIZE);
    imshow("skip", skipped);
    waitKey(0);

    for (int k=1;k<250;k++)
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
            cvtColor(img, img, COLOR_BGR2GRAY);
            // For all optical flow you need a sequence of images.. Or at least 2 of them. Previous
                                    //and current frame
            // if there is no current frame
            // go to this part and fill previous frame
            //else {
            // img.copyTo(prevgray);
            //   }
            // if previous frame is not empty.. There is a picture of previous frame. Do some 
            //optical flow alg. 


            if (prevgray.empty() == false) {
                // calculate optical flow 
                calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
                // copy Umat container to standard Mat
                flowUmat.copyTo(flow);

                flow_fields.push_back(flow);

                Mat colors;

                Mat flow_parts[2];
                split(flow, flow_parts);

                // Convert the algorithm's output into Polar coordinates
                Mat magnitude, angle ;
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
                
                Vec3b color;
                color = colors.at<Vec3b>(Point(250, 300));

                cout << (float)color[0]<< endl;
                Mat whiteimage(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

                for (int y = 0; y < original.rows; y += 5) {
                    for (int x = 0; x < original.cols; x += 10) {
                        if (original.at<Vec3b>(Point2f(x, y)) != Vec3b{0,0,0})
                        {
                            const Point2f flowatxy = flow.at<Point2f>(y, x);
                            // draw line at flow direction
                            color = colors.at<Vec3b>(Point2f(x, y));
                            arrowedLine(whiteimage, Point(x, y), Point((x + flowatxy.x), (y + flowatxy.y)), Scalar((float)color[0], (float)color[1], (float)color[2]));
                            // draw initial point
                            //circle(whiteimage, Point(x, y), 1, Scalar((float)color[0], (float)color[1], (float)color[2]), -1); 
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
    Mat trajectories = convert_to_trajectories(flow_fields, skipped);
    namedWindow("trajectories", WINDOW_AUTOSIZE);
    imshow("trajectories", trajectories);
    waitKey(0);

}