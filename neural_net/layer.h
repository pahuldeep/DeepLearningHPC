#ifndef LAYER_H
#define LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <string>

#include "loss.h"

using namespace std;

class Layer{
public:
    Layer();
    ~Layer();
};


class Dense: public Layer{
public:
    Dense();
    virtual ~Dense();
};


class Activation: public Layer{
public:
    Activation();
    virtual ~Activation();
};


class Softmax: public Layer{
public:
    Softmax();
    virtual ~Softmax();
};



#endif // LAYER_H
