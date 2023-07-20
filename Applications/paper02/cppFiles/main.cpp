#include "functions.h"


int main(int argc, const char** argv)
{
    vector <GpuMat> frames;
    Point position;
    int temp_size;
    int ROI_size;
    GpuMat flow;
    GpuMat velocity;
    int time;
    int v_mean;
    int v_std;
    string folder_path = "F:/SBME 4/GP/CLUST 2015/data/usliverseq/volunteer01";
    frames = readImages(folder_path);
    GpuMat init_temp = getTemplate(frames[0], position, temp_size);

    for (int i=0 ; i < frames.size()-2; i++) {
        GpuMat frame_1 = frames[i];
        GpuMat frame_2 = frames[i + 1];
        GpuMat temp_1, temp_2, ROI_1, ROI_2;
        temp_1 = getTemplate(frame_1, position, temp_size);
        //temp_2 = getTemplate(frame_2, position, temp_size);
        ROI_1 = getROI(frame_1, position, ROI_size);
        ROI_2 = getROI(frame_2, position, ROI_size);
        flow = calcFarnbeck(frame_1, frame_2);
        calcVelocity(flow, time);
        meanStdDev(velocity, v_mean, v_std);
    }
}

// SOME NOTES FROM A FIRST READING
// what's the use for template ? => i don't use it to get ROI ?
// what's the use for ROI ? => i don't use it to get flow field ?