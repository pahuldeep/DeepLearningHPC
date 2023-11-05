#ifndef BLOB_H
#define BLOB_H

#include <array>

namespace deep {

typedef enum{ host, device } DeviceType;

template <typename ptype> class blob{

private:
    ptype *host_pointer = nullptr;
    ptype *device_pointer = nullptr;

    int n = 1;
    int c = 1;
    int h = 1;
    int w = 1;

public:
    blob(int number = 1, int channel = 1, int height = 1, int width = 1): n(number), c(channel), h(height), w(width){
        host_pointer = new float[n * c * h * w];
    }
    blob(std::array<int, 4> size): n(size[0]), c(size[1]), h(size[2]), w(size[3]){
        host_pointer = new float[n * c * h * w];
    }
    ~blob(){
        if(host_pointer != nullptr){
            delete[] host_pointer;
        }
        if(device_pointer != nullptr){
            cudaFree(device_pointer);
        }
    }
};



}

#endif // BLOB_H
