#ifndef LOADER_H
#define LOADER_H

#include <cudnn.h>
#include <cublas_v2.h>

#include <array>

namespace deep {

typedef enum{ host, device } DeviceType;

template <typename dtype>

class load{

public:
    load(int number = 1, int channel = 1, int height = 1, int width = 1): n(number), c(channel), h(height), w(width){
        host_pointer = new float[n * c * h * w];
    }
    load(std::array<int, 4> size): n(size[0]), c(size[1]), h(size[2]), w(size[3]){
        host_pointer = new float[n * c * h * w];
    }
    ~load(){
        if(host_pointer   != nullptr) { delete[] host_pointer; }
        if(device_pointer != nullptr) { cudaFree(device_pointer); }
        if(isTensor)                  { cudnnDestroyTensorDescriptor(tensor_descript); }
    }

    void reset(int number = 1, int channel = 1, int height = 1, int width = 1){

        n = number;
        c = channel;
        h = height;
        w = width;

        if(host_pointer != nullptr){
            delete[] host_pointer;
            host_pointer = nullptr;
        }
        if(device_pointer != nullptr){
            cudaFree(device_pointer);
            device_pointer = nullptr;
        }
        if(isTensor){

        }
    }
    void reset(std::array<int, 4> size){
        reset(size[0], size[1], size[2], size[3]);
    }

    std::array<int, 4> shape(){
        return std::array<int, 4>({n, c, h, w});
    }

    // return number of total elements
    int len(){
        return n * c * h * w;
    }

    // return size
    int buffer_size(){
        return sizeof(dtype) * len();
    }

    // get cpu memory pointer
    dtype *pointer() { return host_pointer; }

    // get gpu memory pointer
    dtype *cuda(){
        if(device_pointer == nullptr){
            cudaMalloc((void**)&device_pointer, sizeof(dtype) * len());
        }
        return device_pointer;
    }

    // transfer data between memory
    dtype *to(DeviceType target){
        dtype *pointer = nullptr;

        if( target == host){
            cudaMemcpy(host_pointer, cuda(), sizeof(dtype)*len(), cudaMemcpyDeviceToHost);
            pointer = host_pointer;
        }else{
            cudaMemcpy(cuda(), host_pointer, sizeof(dtype)*len(), cudaMemcpyHostToDevice);
            pointer = device_pointer;
        }
        return pointer;
    }

    // Tensor control
    bool isTensor = false;
    cudnnTensorDescriptor_t tensor_descript;

    cudnnTensorDescriptor_t tensor(){

        if(isTensor){
            return tensor_descript;
        }

        cudnnCreateTensorDescriptor(&tensor_descript);
        cudnnSetTensor4dDescriptor(tensor_descript, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        isTensor = true;

        return tensor_descript;
    }

private:

    dtype *host_pointer = nullptr;
    dtype *device_pointer = nullptr;

    int n = 1;
    int c = 1;
    int h = 1;
    int w = 1;
};

}

#endif // LOADER_H
