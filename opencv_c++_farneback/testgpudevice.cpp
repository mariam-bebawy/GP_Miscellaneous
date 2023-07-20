#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
using namespace cv;

#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

int main()
{
    printShortCudaDeviceInfo(getDevice());
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
    return 0;
}