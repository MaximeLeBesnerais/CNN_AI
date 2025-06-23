#pragma once

#include "cnn/layer.h"

namespace cnn {

class FlattenLayer : public Layer {
private:
    std::vector<int> original_shape_;
    
public:
    FlattenLayer() = default;
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override { return "Flatten"; }
};

class BatchNormLayer : public Layer {
private:
    int num_features_;
    float eps_;
    float momentum_;
    
    // Learnable parameters
    Tensor gamma_;      // Scale parameter
    Tensor beta_;       // Shift parameter
    
    // Gradients
    Tensor gamma_grad_;
    Tensor beta_grad_;
    
    // Running statistics (for inference)
    Tensor running_mean_;
    Tensor running_var_;
    
    // Cached values for backward pass
    Tensor cached_input_;
    Tensor cached_mean_;
    Tensor cached_var_;
    Tensor cached_normalized_;
    
public:
    BatchNormLayer(int num_features, float eps = 1e-5f, float momentum = 0.1f);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "BatchNorm"; }
    
    std::vector<Tensor*> get_parameters() override;
    std::vector<Tensor*> get_gradients() override;
    
    // Accessors
    const Tensor& gamma() const { return gamma_; }
    const Tensor& beta() const { return beta_; }
    const Tensor& running_mean() const { return running_mean_; }
    const Tensor& running_var() const { return running_var_; }
};

} // namespace cnn
