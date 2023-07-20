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

using namespace cv;
using namespace std;

Mat convert_to_trajectories(vector<Mat> flowfields) {
	// flowfields is a Mat of shape rows*cols*time
	//initialize trajectories with a white image
    Mat initial_flow = flowfields[0];
	Mat trajectories(initial_flow.rows, initial_flow.cols, CV_8UC3, Scalar(255, 255, 255));
	
	//loop over each pixel and follow its path
    
    for (int y = 0; y < initial_flow.rows; y += 5) {
        for (int x = 0; x < initial_flow.cols; x += 10) {
            Point prev = Point(y, x);
            Point current;
            for (int i = 0; i < flowfields.size(); i++) {
                Point displacement = flowfields[i].at<Point2f>(prev.y, prev.x);
                current = Point(cvRound(prev.x + displacement.x), cvRound(prev.y + displacement.y));
                line(trajectories, prev, current, Scalar(i, i, 255 - i));
            }
        }
    }
	return trajectories;
}