//    TODO
//    1. First, perform forward operation.
//    2. Then,  perform backward operation.
//    3. Then,  get a weight update from the gradient.
//    4. Finally, the output layer will obtain the loss.

#include <cuda_runtime.h>
#include <curand.h>

#include "layer.h"

using namespace deep;


//**************************************************************
// Layer
//**************************************************************

Layer::Layer(){
}

Layer::~Layer(){
}

//**************************************************************
// Dense Layer
//**************************************************************

Dense::Dense(std::string names, int outputSize){
    name = names;
    output_size = outputSize;
}

load<float> *Dense::forward(load<float> *input)
{
    // Initialize output loader

    // output = weight^T * input (withou bias)
    cublasSgemm(cuda->cublas(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                output_size, batch_size, input_size,
                &cuda->positive_one,
                weight->cuda(), input_size,
                input->cuda(), input_size,
                &cuda->zero,
                output->cuda(), output_size );

    // output += biases * d_one_vec^T

}




//**************************************************************
// Activation
//**************************************************************



//**************************************************************
// Softmax
//**************************************************************
