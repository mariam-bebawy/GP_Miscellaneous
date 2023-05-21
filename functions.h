#pragma once

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
#include<tuple>
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
#include <fstream>
#include <opencv2/cudaarithm.hpp>



using namespace std;
using namespace cv;
using namespace cv::cuda;
//#include"Conversion_from_flowfields _to _trajectories.cpp"

//create point of position Xk-1
//function to read images in vector <mat> 
vector <GpuMat> readImages(String Path);
//get template of size Stemplate
GpuMat getTemplate(GpuMat frame, Point position, int size );
//get ROI of the liver of size 3*Stemplate
GpuMat getROI(GpuMat frame, Point position, int size);
//calc optical flow between two ROIs and return flow
GpuMat calcFarnbeck(GpuMat frame1, GpuMat frame2);
//calc velocity matrix with the same size of ROI and -->>> remember to calc mean and std outside of functions -->> note that there is a sanity check for velocity
void calcVelocity(GpuMat flow, int time);
//created the Gaussian weighted map given the position Xk-1, mean v, v std
GpuMat calcGaussianMap(Point Xk_1, Scalar mean, Scalar std);
// weighting map based on normalized cross - correlation, nalt the number of alternative templates, walt the weighting of the alterna
// remember to invert the current frame before M_im according to "intensity at the desired point is lower than the mean intensity throughout the region encompassed by the initial template"
GpuMat calcWeightingMap(int w_alt, vector<GpuMat> alt_temps, GpuMat init_frame, GpuMat curr_frame);
//normalize Ms
GpuMat normalizeMaps(GpuMat weighting_map);
//get the predicted position knowing normalizaed M_corr, M_vel , M_im
Point getFinalPosition(GpuMat M_final );
//plot point 
void showResult(Mat frame);

