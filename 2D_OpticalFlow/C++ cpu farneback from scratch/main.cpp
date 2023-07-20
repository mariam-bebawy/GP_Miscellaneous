#include "Header.h"

int main(int argc, const char** argv)
{

    string path = "videos/dna.avi";

    
    VideoCapture cap(path);
    double frame_Number = cap.get(CAP_PROP_FRAME_COUNT);
    cout << frame_Number << endl;

    bool Is = cap.grab();
    Mat frame, prevgray;

    cap.retrieve(frame, CAP_OPENNI_GRAY_IMAGE);
    cv::resize(frame, frame, Size(40, 40), INTER_LINEAR);


    cout << frame.rows << "  " << frame.cols << endl;

    Mat dx(Size(frame.cols, frame.rows), CV_32F);
    Mat dy(Size(frame.cols, frame.rows), CV_32F);

    vector<Mat>Dx;
    vector<Mat>Dy;

    while (Is) {
        cv::cvtColor(frame, frame, COLOR_BGR2GRAY); 
        frame.convertTo(frame, CV_32F, 1.0 / 255, 0);
        //cv::GaussianBlur(frame, frame, Size(7, 7), 1.0);
        cv::resize(frame, frame, Size(40, 40), INTER_LINEAR);

        if (prevgray.empty() == false) {


            calc_flow(frame, prevgray,dx,dy);
            Dx.push_back(dx);
            Dy.push_back(dy);

            visualize_flow(dx,dy);
        }
        frame.copyTo(prevgray);
        Is = cap.grab();
        cap.retrieve(frame, CAP_OPENNI_GRAY_IMAGE);

    }


}