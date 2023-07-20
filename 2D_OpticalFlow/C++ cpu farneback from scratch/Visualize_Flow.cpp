#include "Header.h"

void visualize_flow(Mat &dx,Mat &dy) {
    Mat colors;

    // Convert the algorithm's output into Polar coordinates
    Mat magnitude, angle;
    cv::cartToPolar(dx, dy, magnitude, angle, true);
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
    Mat whiteimage(dx.rows, dx.cols, CV_8UC3, Scalar(255, 255, 255));
    Point2f flowatxy;
    for (int y = 0; y < dx.rows; y += 4) {
        for (int x = 0; x < dx.rows; x += 8) {
            flowatxy = Point2f(dx.at<float>(y, x), dy.at<float>(y, x)) * 10;
            color = colors.at<Vec3b>(Point2f(x, y));
            if (!(flowatxy.x == 0 && flowatxy.y == 0))
                arrowedLine(whiteimage, Point(x, y), Point((x + flowatxy.x), (y + flowatxy.y)), Scalar((float)color[0], (float)color[1], (float)color[2]));
        }
    }


    namedWindow("flow", WINDOW_AUTOSIZE);
    imshow("flow", whiteimage);
    waitKey(10);

}









