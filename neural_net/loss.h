#ifndef LOSS_H
#define LOSS_H

#include "loader.h"

namespace deep {

class CrossEntropyLoss{

public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    float loss();
    float accuracy();
};


}


#endif // LOSS_H
