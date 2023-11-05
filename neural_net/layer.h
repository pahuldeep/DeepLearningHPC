#ifndef LAYER_H
#define LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <string>

#include "blob.h"
#include "loss.h"

namespace deep{

class Layer{

public:
    Layer();
    ~Layer();

    virtual blob<float> *forward(blob<float> *input) = 0;
    virtual blob<float> *backward(blob<float> *grad_input) = 0;

    std::string get_name() { return name; }

    virtual float get_loss(blob<float> *target);
    virtual int get_accuracy(blob<float> *target);

    void set_cuda_context(cudnnContext *context) {cuda = context;}

    void freeze()  { freeze_ = true;}
    void unfreeze(){ freeze_ = false;}
    void set_load_pretrain() { load_pretrain = true; }
    void set_gradient_stop() { gradient_stop = true; }

protected:
    std::string name;

    // Tensor input/output
    cudnnTensorDescriptor_t input_descript;
    cudnnTensorDescriptor_t output_descript;

    // filter/bias for weights and biases
    cudnnTensorDescriptor_t bias_descript;
    cudnnFilterDescriptor_t filter_descript;

    // output memory
    blob<float> *input       = nullptr;    /* x  */
    blob<float> *output      = nullptr;    /* y  */
    blob<float> *grad_input  = nullptr;    /* dx */
    blob<float> *grad_output = nullptr;    /* dy */

    // master weights & bias
    bool freeze_               = false;     /* control parameter updates */

    blob<float> *weights      = nullptr;   /* w */
    blob<float> *biases       = nullptr;   /* b */
    blob<float> *grad_weights = nullptr;   /* dw */
    blob<float> *grad_biases  = nullptr;   /* db */

    int batch_size = 0;

    // initialize weights along with the input size
    void init_weight_bias(unsigned int seed = 0);
    void update_weights_biases(float learning_rate);

    cudnnContext *cuda = nullptr;

    friend class Network;

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

}

#endif // LAYER_H
