// Copyright 2020-present pytorch-cpp Authors
#include "neural_net.h"
#include <torch/torch.h>

ConvNetImpl::ConvNetImpl(int64_t num_classes)
    : fc1(3 * 26 * 26, 4096), fc2(4096, num_classes) {
    register_module("layer1", layer1);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
    x = layer1->forward(x);
    x = x.view({-1, 3 * 26 * 26});
    // x = x.view({-1,  64 * 4 * 4});
    x = torch::nn::functional::relu(fc1->forward(x));
    return fc2->forward(x);
}
