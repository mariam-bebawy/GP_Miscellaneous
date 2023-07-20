#include "functions.h"

void visualizetrajectories(vector<Mat>xp, vector<Mat>yp, Mat* trajectories_img){


    int rows = xp[0].rows;
    int cols = xp[0].cols;
    int t = xp.size();

    Point2f current, prev;

    for (int y = 0; y < rows; y += 1) {
        for (int x = 0; x < cols; x += 1) {
            prev = Point2f(x, y)*10.0;
            for (int i = 0; i < t - 1; i++) {
                current = Point2f(xp[i + 1].at<float>(y, x), yp[i + 1].at<float>(y, x))*10.0;
                if (Point(prev.x,prev.y)!=Point(current.x,current.y)) {
                    arrowedLine(*trajectories_img, prev, current, Scalar(220-i, 100+0.75*i ,  0.9*i));
                }
                prev = current;

            }
        }
    } 

    /*int start = (cols / 2 - cols / 5) * 10;
    int end = start + 2550;
    int y_start = rows * 10;
    int y_end = y_start - 200;

    for (int x = start; x < end; x += 1) {
        for (int y = x; y < x + 10; y++) {
            arrowedLine(*trajectories_img, Point(y, y_start), Point(y, y_end), Scalar(220- 0.1*(x - start), 100 + 0.075 * (x - start), 0.09*(x - start)));

        }
    }*/
}
