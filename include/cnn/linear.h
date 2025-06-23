#pragma once

#include "cnn/layer.h"

namespace cnn {

class LinearLayer : public Layer {
private:
    int input_size_;
    int output_size_;
    
    Tensor weights_;      // Shape: (input_size, output_size)
    Tensor bias_;         // Shape: (output_size,)
    
    Tensor weight_grad_;  // Gradient for weights
    Tensor bias_grad_;    // Gradient for bias
    
    Tensor cached_input_; // Cache input for backward pass
    
public:
    LinearLayer(int input_size, int output_size);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "Linear"; }
    
    std::vector<Tensor*> get_parameters() override;
    std::vector<Tensor*> get_gradients() override;
    
    // Weight initialization methods
    void xavier_init();
    void he_init();
    void zero_init();
    void random_init(float std = 0.01f);
    
    // Accessors
    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }
    Tensor& weights() { return weights_; }
    Tensor& bias() { return bias_; }
    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }
};

} // namespace cnn
