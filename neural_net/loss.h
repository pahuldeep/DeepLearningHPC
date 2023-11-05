#ifndef LOSS_H
#define LOSS_H

class CrossEntropyLoss{
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    float loss();
    float accuracy();
};

#endif // LOSS_H
