// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
 public:
    explicit ConvNetImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 3, 3).stride(1)),
        torch::nn::ReLU(),
    };

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

TORCH_MODULE(ConvNet);
