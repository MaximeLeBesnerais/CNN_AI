#pragma once

#include "cnn/layer.h"

namespace cnn {

class Conv2DLayer : public Layer {
private:
    int in_channels_;
    int out_channels_;
    int kernel_height_;
    int kernel_width_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    bool use_bias_;
    
    // Parameters
    Tensor weights_;      // Shape: (out_channels, in_channels, kernel_h, kernel_w)
    Tensor bias_;         // Shape: (out_channels,)
    
    // Gradients
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    // Cached values for backward pass
    Tensor cached_input_;
    Tensor cached_im2col_;
    std::vector<int> output_shape_;
    
public:
    Conv2DLayer(int in_channels, int out_channels, 
                int kernel_size, int stride = 1, int padding = 0, bool use_bias = true);
    
    Conv2DLayer(int in_channels, int out_channels,
                int kernel_height, int kernel_width,
                int stride_h = 1, int stride_w = 1,
                int pad_h = 0, int pad_w = 0, bool use_bias = true);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "Conv2D"; }
    
    std::vector<Tensor*> get_parameters() override;
    std::vector<Tensor*> get_gradients() override;
    
    // Weight initialization
    void xavier_init();
    void he_init();
    void random_init(float std = 0.01f);
    
    // Accessors
    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }
    int in_channels() const { return in_channels_; }
    int out_channels() const { return out_channels_; }
    int kernel_height() const { return kernel_height_; }
    int kernel_width() const { return kernel_width_; }
    
private:
    void init_parameters();
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const;
};

} // namespace cnn
