#ifndef NETWORK_H
#define NETWORK_H

#include "loss.h"
#include "layer.h"

typedef enum{
    training,
    inference
} workload;

class Network{
public:
    Network();
    ~Network();

    void add_layer(Layer *layer);

    // forward @declare blob

    void backward();
    void update();

    void loss();
    void get_accuracy();

    void train();
    void test();
    void cuda_compile();

};

#endif // NETWORK_H
