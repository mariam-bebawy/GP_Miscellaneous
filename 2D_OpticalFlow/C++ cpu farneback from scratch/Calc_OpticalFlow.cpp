#include "Header.h"

using namespace std;
void calc_flow(Mat &f1, Mat &f2,Mat &dx , Mat& dy) {

    int rows = f1.rows;
    int cols = f1.cols;


    Mat c1(rows,cols, CV_32F);
    Mat c2(rows, cols, CV_32F);


    create_CertaintyMatrix(rows, cols,c1);
    create_CertaintyMatrix(rows, cols,c2);



    int n_pyr = 4;
    int sigma = 4;
    int sigma_flow = 4;
    int num_iter = 3;




   
    vector<Mat> pyramidframe1(n_pyr);
    vector<Mat> pyramidframe2(n_pyr);
    vector<Mat> pyramidc1(n_pyr);
    vector<Mat> pyramidc2(n_pyr);
    vector<Mat> pyramiddx(n_pyr);
    vector<Mat> pyramiddy(n_pyr);

    f1.copyTo(pyramidframe1[0]);
    f2.copyTo(pyramidframe2[0]);
    c1.copyTo(pyramidc1[0]);
    c2.copyTo(pyramidc2[0]);
    dx.copyTo(pyramiddx[0]);
    dy.copyTo(pyramiddy[0]);


   
   
    for (int i = 0; i < n_pyr-1; i++) {
        pyrDown(pyramidframe1[i], pyramidframe1[i+1], Size(pyramidframe1[i].cols / 2, pyramidframe1[i].rows / 2));
        pyrDown(pyramidframe2[i], pyramidframe2[i+1], Size(pyramidframe2[i].cols / 2, pyramidframe2[i].rows / 2));
        pyrDown(pyramidc1[i], pyramidc1[i+1], Size(pyramidc1[i].cols / 2, pyramidc1[i].rows / 2));
        pyrDown(pyramidc2[i], pyramidc2[i+1], Size(pyramidc2[i].cols / 2, pyramidc2[i].rows / 2));
        pyrDown(pyramiddy[i], pyramiddy[i + 1], Size(pyramiddy[i].cols / 2, pyramiddy[i].rows / 2));
        pyrDown(pyramiddx[i], pyramiddx[i + 1], Size(pyramiddx[i].cols / 2, pyramiddx[i].rows / 2));

    }


    Mat dx_current;
    Mat dy_current;
    for (int i = n_pyr-1; i >=0; i--) {
        dx_current = pyramiddx[i];
        dy_current = pyramiddy[i];
        
        if (i == n_pyr - 1) {


            flow_Iterative(pyramidframe1[i], pyramidframe2[i], pyramidc1[i], pyramidc2[i], sigma, sigma_flow, dx_current, dy_current, true);

        }
        else {
            pyrUp(pyramiddx[i+1], pyramiddx[i], Size(pyramiddx[i].cols , pyramiddx[i].rows ));
            pyrUp(pyramiddy[i+1], pyramiddy[i], Size(pyramiddy[i].cols , pyramiddy[i].rows ));
            flow_Iterative(pyramidframe1[i], pyramidframe2[i], pyramidc1[i], pyramidc2[i], sigma, sigma_flow, pyramiddx[i], pyramiddy[i], false);

        }
        cout << "flow iterative of pyr = " << i << endl;

    }

    pyramiddx[0].copyTo(dx);
    pyramiddy[0].copyTo(dy);


    

}