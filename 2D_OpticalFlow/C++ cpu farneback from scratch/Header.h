#ifndef functions
#define functions

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
#include <opencv2/cudacodec.hpp>
#include <omp.h>
#include <chrono>
#include <time.h>
#include<tuple>
#include<opencv2/core/cuda.hpp>
#include<Eigen/Dense>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;
using Eigen::MatrixXd;


void polynomialExpansionCoeff(Mat& f, Mat& c, float sigma, vector<vector<Eigen::MatrixXf>>& A, vector<Eigen::MatrixXf>& B, Eigen::MatrixXf& C);

void flow_Iterative(Mat &f1, Mat &f2, Mat &c1, Mat &c2, float sigma, float sigma_flow, Mat &dx, Mat &dy, bool initial);

void calc_flow(Mat& f1, Mat& f2, Mat& dx,  Mat& dy);
void visualize_flow(Mat& dx, Mat& dy);
void create_CertaintyMatrix(int rows, int cols, Mat& c);
void correlate1d(vector<float> input, vector<float> filter, vector<float>& result);
int check_bounds(int x, int max);




#endif#include <iostream>