#ifndef HELPER_H
#define HELPER_H

#include <cudnn.h>
#include <cublas_v2.h>

#include <curand.h>
namespace deep {

// container for cuda resources
class CudaContext{

public:
    CudaContext(){
        cublasCreate(&cublas_handle);

        cudaGetLastError();
        cudnnCreate(&cudnn_handle);
    }
    ~CudaContext(){

        cublasDestroy(cublas_handle);
        cudnnDestroy(cudnn_handle);
    }

    cublasHandle_t cublas(){
        return cublas_handle;
    }
    cudnnHandle_t  cudnn(){
        return cudnn_handle;
    }

    const float positive_one = 1.f;
    const float zero = 0.f;
    const float negative_one = -1.f;

private:
    cublasHandle_t cublas_handle;
    cudnnHandle_t  cudnn_handle;
};

}


#endif // HELPER_H
