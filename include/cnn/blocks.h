#pragma once

#include "cnn/layer.h"
#include "cnn/conv2d.h"
#include "cnn/activation.h"
#include "cnn/layers.h"
#include "cnn/pooling.h"
#include "cnn/tensor.h"
#include <vector>
#include <memory>

namespace cnn {

// Forward declarations
class Conv2DLayer;
class BatchNormLayer;
class ReLULayer;

/**
 * @brief Depthwise Convolution Layer
 * 
 * Performs depthwise separable convolution where each input channel
 * is convolved with its own set of filters. More efficient than regular
 * convolution for mobile/embedded applications.
 */
class DepthwiseConv2DLayer : public Layer {
private:
    int in_channels_;
    int kernel_height_, kernel_width_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    
    Tensor weights_;  // Shape: [in_channels, 1, kernel_h, kernel_w]
    Tensor bias_;     // Shape: [in_channels]
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    Tensor last_input_;  // Cache for backward pass
    
public:
    DepthwiseConv2DLayer(int in_channels, int kernel_height, int kernel_width,
                        int stride_h = 1, int stride_w = 1,
                        int pad_h = 0, int pad_w = 0);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "DepthwiseConv2D"; }
    
private:
    void initialize_weights();
    Tensor depthwise_conv_forward(const Tensor& input);
    Tensor depthwise_conv_backward_input(const Tensor& output_grad);
    void depthwise_conv_backward_weights(const Tensor& input, const Tensor& output_grad);
};

/**
 * @brief Residual Block
 * 
 * Implements a basic residual block with skip connection.
 * Input -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU
 *   |                                      ^
 *   +--------------------------------------+
 */
class ResidualBlock : public Layer {
private:
    std::unique_ptr<Conv2DLayer> conv1_;
    std::unique_ptr<BatchNormLayer> bn1_;
    std::unique_ptr<ReLULayer> relu1_;
    std::unique_ptr<Conv2DLayer> conv2_;
    std::unique_ptr<BatchNormLayer> bn2_;
    std::unique_ptr<ReLULayer> relu2_;
    
    // Optional 1x1 conv for dimension matching
    std::unique_ptr<Conv2DLayer> shortcut_conv_;
    std::unique_ptr<BatchNormLayer> shortcut_bn_;
    
    Tensor last_input_;  // Cache for skip connection
    bool use_shortcut_;
    
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "ResidualBlock"; }
};

/**
 * @brief Inception Module
 * 
 * Implements basic Inception module with multiple parallel paths:
 * - 1x1 conv
 * - 1x1 -> 3x3 conv
 * - 1x1 -> 5x5 conv
 * - 3x3 maxpool -> 1x1 conv
 * All outputs are concatenated along channel dimension.
 */
class InceptionModule : public Layer {
private:
    // Branch 1: 1x1 conv
    std::unique_ptr<Conv2DLayer> branch1_conv_;
    
    // Branch 2: 1x1 -> 3x3 conv
    std::unique_ptr<Conv2DLayer> branch2_conv1_;
    std::unique_ptr<Conv2DLayer> branch2_conv2_;
    
    // Branch 3: 1x1 -> 5x5 conv
    std::unique_ptr<Conv2DLayer> branch3_conv1_;
    std::unique_ptr<Conv2DLayer> branch3_conv2_;
    
    // Branch 4: maxpool -> 1x1 conv
    std::unique_ptr<MaxPoolingLayer> branch4_pool_;
    std::unique_ptr<Conv2DLayer> branch4_conv_;
    
    // ReLU activations for each branch
    std::unique_ptr<ReLULayer> relu1_, relu2a_, relu2b_, relu3a_, relu3b_, relu4_;
    
    int in_channels_;
    int branch1_channels_, branch2_channels_, branch3_channels_, branch4_channels_;
    
public:
    InceptionModule(int in_channels, 
                   int branch1_channels = 64,
                   int branch2_channels = 96, 
                   int branch3_channels = 16,
                   int branch4_channels = 32);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "InceptionModule"; }
    
private:
    Tensor concatenate_channels(const std::vector<Tensor>& tensors);
    std::vector<Tensor> split_channels(const Tensor& tensor, const std::vector<int>& channel_sizes);
};

/**
 * @brief Bottleneck Block
 * 
 * Implements bottleneck block commonly used in ResNet and other architectures:
 * 1x1 conv (reduce) -> 3x3 conv -> 1x1 conv (expand)
 * Each conv is followed by BatchNorm and ReLU (except last ReLU is after skip connection)
 */
class BottleneckBlock : public Layer {
private:
    std::unique_ptr<Conv2DLayer> conv1_;  // 1x1 reduce
    std::unique_ptr<BatchNormLayer> bn1_;
    std::unique_ptr<ReLULayer> relu1_;
    
    std::unique_ptr<Conv2DLayer> conv2_;  // 3x3 conv
    std::unique_ptr<BatchNormLayer> bn2_;
    std::unique_ptr<ReLULayer> relu2_;
    
    std::unique_ptr<Conv2DLayer> conv3_;  // 1x1 expand
    std::unique_ptr<BatchNormLayer> bn3_;
    
    // Optional shortcut
    std::unique_ptr<Conv2DLayer> shortcut_conv_;
    std::unique_ptr<BatchNormLayer> shortcut_bn_;
    
    std::unique_ptr<ReLULayer> final_relu_;
    
    Tensor last_input_;
    bool use_shortcut_;
    
public:
    BottleneckBlock(int in_channels, int bottleneck_channels, int out_channels, int stride = 1);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    void update_weights(float learning_rate) override;
    std::string name() const override { return "BottleneckBlock"; }
};

} // namespace cnn
