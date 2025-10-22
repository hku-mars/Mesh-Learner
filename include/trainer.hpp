#ifndef __IMMESH_RENDERER_TRAINER_HPP__
#define __IMMESH_RENDERER_TRAINER_HPP__

#include <stdlib.h>
#include <stdio.h>
#include <torch/torch.h>
#include "include/MeshLearner.cuh"

using namespace torch::autograd;
using namespace std;

namespace MeshLearner {

class MyTrainer : public Function<MyTrainer> {
public:
    static torch::Tensor forward(AutogradContext *ctx, at::Tensor curTrainShTex, unsigned int* errCnt, int shLayerNum, int shOrders, int imgW, int imgH);

    static tensor_list backward(AutogradContext *ctx, tensor_list gradOutPuts);
};

}

#endif