#ifndef NETWORK_H
#define NETWORK_H

#include "loss.h"
#include "layer.h"
#include "helper.h"

namespace deep {

typedef enum{
    training,
    inference
} workload;

class Network{
public:
    Network();
    ~Network();

    void add_layer(Layer *layer);

    load<float> *forward(load<float> *input);
    void backward(load<float> *input = nullptr);
    void update(float learning_rate = 0.02f);

    void loss(load<float> *target);
    void get_accuracy(load<float> *target);

    void train();
    void test();
    void cuda_compile();

};

}


#endif // NETWORK_H
