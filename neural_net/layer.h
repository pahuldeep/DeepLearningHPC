#ifndef LAYER_H
#define LAYER_H

#include <cublas_v2.h>
#include <cudnn.h>

#include <string>

#include "loader.h"
#include "helper.h"

namespace deep{

class Layer{

public:
    Layer();
    ~Layer();

    virtual load<float> *forward(load<float>  *input) = 0;
    virtual load<float> *backward(load<float> *grad_input) = 0;

    std::string get_name() { return name; }

    virtual float get_loss(load<float>   *target);
    virtual int get_accuracy(load<float> *target);

    void set_cuda_context(CudaContext *context) { cuda = context; }

    void freeze()  { Freeze = true;}
    void unfreeze(){ Freeze = false;}

    void set_load_pretrain() { load_pretrain = true; }
    void set_gradient_stop() { gradient_stop = true; }

protected:
    virtual void forward_init(load<float> *intput) = 0;
    virtual void backward_init(load<float> *grad_output) = 0;

    std::string name;

    // Tensor input/output
    cudnnTensorDescriptor_t input_descript;
    cudnnTensorDescriptor_t output_descript;

    // filter/bias for weights and biases
    cudnnTensorDescriptor_t bias_descript;
    cudnnFilterDescriptor_t filter_descript;

    // output memory
    load<float> *input       = nullptr;    /* x  */
    load<float> *output      = nullptr;    /* y  */
    load<float> *grad_input  = nullptr;    /* dx */
    load<float> *grad_output = nullptr;    /* dy */

    // master weights & bias
    bool Freeze              = false;     /* control parameter updates */

    load<float> *weight      = nullptr;   /* w */
    load<float> *bias        = nullptr;   /* b */
    load<float> *grad_weight = nullptr;   /* dw */
    load<float> *grad_biase  = nullptr;   /* db */

    int batch_size = 0;

    // initialize weights along with the input size
    void init_weight(unsigned int seed = 0);
    void update_weight(float learning_rate);

    CudaContext *cuda = nullptr;

    bool load_pretrain = false;
    bool gradient_stop = false;

    int load_parameter();
    int save_parameter();

    friend class Network;

};


class Dense: public Layer{
public:
    Dense(std::string name, int size);
    virtual ~Dense();

    virtual load<float> *forward(load<float>  *input);
    virtual load<float> *backward(load<float> *grad_input);

private:
    void forward_init(load<float> *input);
    void backward_init(load<float> *grad_output);

    int input_size = 0;
    int output_size = 0;

    float *device_oneVector = nullptr;

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

}

#endif // LAYER_H
