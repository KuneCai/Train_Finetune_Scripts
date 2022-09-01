// Copyright 2020-present pytorch-cpp Authors
#include "neural_net.h"
#include <torch/torch.h>

NeuralNetImpl::NeuralNetImpl(int64_t input_size, int64_t num_classes)
    : fc1(input_size, 2048), fc2(2048, 4096) , fc3(4096, num_classes) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
    x = torch::nn::functional::relu(fc1->forward(x));
    x = torch::nn::functional::relu(fc2->forward(x));
    return fc3->forward(x);
}
