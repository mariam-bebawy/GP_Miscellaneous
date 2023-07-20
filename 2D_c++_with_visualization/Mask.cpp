#include "functions.h"


void createMasks (Mat img,Mat &Mask,Mat &Mask_nuc) {
    Mat im_th;
    cv::threshold(img, im_th, 120, 255, THRESH_BINARY);

    // Floodfill from point (0, 0)
    Mat im_floodfill = im_th.clone();
    floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));
    imshow("mask", im_floodfill);

    // Invert floodfilled image
    Mat im_floodfill_inv;
    cv::bitwise_not(im_floodfill, im_floodfill_inv);
    imshow("mask", im_floodfill_inv);

    // Combine the two images to get the foreground.
    Mat Masked;
    Mat im_out = (im_th | im_floodfill_inv);
    //cv::bitwise_and(img, im_out, Masked);
    cv::cvtColor(im_out, im_out, COLOR_BGR2GRAY);
    cv::threshold(im_out, Mask, 0, 255, THRESH_OTSU);
//    imshow("mask", Mask);
    cv::cvtColor(img, img, COLOR_BGR2GRAY);

    cv::bitwise_and(img, Mask, Masked);


    cv::threshold(Masked, Mask_nuc, 110, 255, THRESH_BINARY);
   /* int morph_size = 2;
    Mat element = getStructuringElement(
        MORPH_RECT, Size(2 * morph_size + 1,
            2 * morph_size + 1),
        Point(morph_size, morph_size));
    Mat erod, dill;

    dilate(Masked, Masked, element, Point(-1, -1), 2);

    erode(Masked, Masked, element,Point(-1, -1), 2);*/
    //Mask.convertTo(Mask, CV_32F);
    //Mask_nuc.convertTo(Mask_nuc, CV_32F);

    imshow("nuc_mask", Mask_nuc);

}