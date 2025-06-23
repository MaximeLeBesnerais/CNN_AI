#pragma once

#include "cnn/layer.h"

namespace cnn {

class ReLULayer : public Layer {
private:
    Tensor cached_input_;
    
public:
    ReLULayer() = default;
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "ReLU"; }
};

class LeakyReLULayer : public Layer {
private:
    float alpha_;
    Tensor cached_input_;
    
public:
    LeakyReLULayer(float alpha = 0.01f) : alpha_(alpha) {}
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "LeakyReLU"; }
    
    float alpha() const { return alpha_; }
};

class SigmoidLayer : public Layer {
private:
    Tensor cached_output_;
    
public:
    SigmoidLayer() = default;
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "Sigmoid"; }
};

class TanhLayer : public Layer {
private:
    Tensor cached_output_;
    
public:
    TanhLayer() = default;
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "Tanh"; }
};

class SoftmaxLayer : public Layer {
private:
    Tensor cached_output_;
    int axis_;
    
public:
    SoftmaxLayer(int axis = -1) : axis_(axis) {}
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "Softmax"; }
    
    int axis() const { return axis_; }
};

} // namespace cnn
