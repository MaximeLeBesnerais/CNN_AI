#pragma once

#include "cnn/layer.h"
#include <random>

namespace cnn {

class DropoutLayer : public Layer {
private:
    float dropout_rate_;
    Tensor mask_;
    mutable std::mt19937 generator_;
    mutable std::uniform_real_distribution<float> distribution_;
    
public:
    DropoutLayer(float dropout_rate = 0.5f);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "Dropout"; }
    
    float dropout_rate() const { return dropout_rate_; }
    void set_dropout_rate(float rate);
};

} // namespace cnn
