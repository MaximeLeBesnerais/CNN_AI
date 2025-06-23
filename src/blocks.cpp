#include "cnn/blocks.h"
#include "cnn/conv2d.h"
#include "cnn/activation.h"
#include "cnn/layers.h"
#include "cnn/pooling.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace cnn {

// ============================================================================
// DepthwiseConv2DLayer Implementation
// ============================================================================

DepthwiseConv2DLayer::DepthwiseConv2DLayer(int in_channels, int kernel_height, int kernel_width,
                                         int stride_h, int stride_w, int pad_h, int pad_w)
    : in_channels_(in_channels), kernel_height_(kernel_height), kernel_width_(kernel_width),
      stride_h_(stride_h), stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w) {
    
    initialize_weights();
}

void DepthwiseConv2DLayer::initialize_weights() {
    // Initialize weights with He initialization for ReLU
    int fan_in = kernel_height_ * kernel_width_;
    float std = std::sqrt(2.0f / fan_in);
    
    weights_ = Tensor({in_channels_, 1, kernel_height_, kernel_width_});
    bias_ = Tensor({in_channels_});
    
    // Initialize with random values
    float* weights_data = weights_.data();
    float* bias_data = bias_.data();
    
    for (int i = 0; i < weights_.size(); ++i) {
        weights_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    
    for (int i = 0; i < bias_.size(); ++i) {
        bias_data[i] = 0.0f;
    }
    
    weight_grad_ = Tensor(weights_.shape());
    bias_grad_ = Tensor(bias_.shape());
}

Tensor DepthwiseConv2DLayer::forward(const Tensor& input, bool training) {
    last_input_ = input;
    return depthwise_conv_forward(input);
}

Tensor DepthwiseConv2DLayer::depthwise_conv_forward(const Tensor& input) {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("DepthwiseConv2D expects 4D input (batch, channels, height, width)");
    }
    
    int batch_size = input_shape[0];
    int in_height = input_shape[2];
    int in_width = input_shape[3];
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * pad_h_ - kernel_height_) / stride_h_ + 1;
    int out_width = (in_width + 2 * pad_w_ - kernel_width_) / stride_w_ + 1;
    
    Tensor output({batch_size, in_channels_, out_height, out_width});
    
    // Perform depthwise convolution
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels_; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    // Convolve with the single filter for this channel
                    for (int kh = 0; kh < kernel_height_; ++kh) {
                        for (int kw = 0; kw < kernel_width_; ++kw) {
                            int ih = oh * stride_h_ - pad_h_ + kh;
                            int iw = ow * stride_w_ - pad_w_ + kw;
                            
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                sum += input.at(b, c, ih, iw) * weights_.at(c, 0, kh, kw);
                            }
                        }
                    }
                    
                    output.at(b, c, oh, ow) = sum + bias_.at(c);
                }
            }
        }
    }
    
    return output;
}

Tensor DepthwiseConv2DLayer::backward(const Tensor& output_gradient) {
    // Compute gradients for weights and bias
    depthwise_conv_backward_weights(last_input_, output_gradient);
    
    // Compute gradient w.r.t. input
    return depthwise_conv_backward_input(output_gradient);
}

Tensor DepthwiseConv2DLayer::depthwise_conv_backward_input(const Tensor& output_grad) {
    auto input_shape = last_input_.shape();
    Tensor input_grad(input_shape);
    
    int batch_size = input_shape[0];
    int in_height = input_shape[2];
    int in_width = input_shape[3];
    
    auto output_shape = output_grad.shape();
    int out_height = output_shape[2];
    int out_width = output_shape[3];
    
    // Compute input gradient using transposed convolution
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels_; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float grad_val = output_grad.at(b, c, oh, ow);
                    
                    for (int kh = 0; kh < kernel_height_; ++kh) {
                        for (int kw = 0; kw < kernel_width_; ++kw) {
                            int ih = oh * stride_h_ - pad_h_ + kh;
                            int iw = ow * stride_w_ - pad_w_ + kw;
                            
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                input_grad.at(b, c, ih, iw) += grad_val * weights_.at(c, 0, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return input_grad;
}

void DepthwiseConv2DLayer::depthwise_conv_backward_weights(const Tensor& input, const Tensor& output_grad) {
    // Zero gradients
    float* weight_grad_data = weight_grad_.data();
    float* bias_grad_data = bias_grad_.data();
    
    // Zero out gradients
    for (int i = 0; i < weight_grad_.size(); ++i) {
        weight_grad_data[i] = 0.0f;
    }
    for (int i = 0; i < bias_grad_.size(); ++i) {
        bias_grad_data[i] = 0.0f;
    }
    
    auto input_shape = input.shape();
    auto output_shape = output_grad.shape();
    
    int batch_size = input_shape[0];
    int in_height = input_shape[2];
    int in_width = input_shape[3];
    int out_height = output_shape[2];
    int out_width = output_shape[3];
    
    // Compute weight gradients
    for (int c = 0; c < in_channels_; ++c) {
        for (int kh = 0; kh < kernel_height_; ++kh) {
            for (int kw = 0; kw < kernel_width_; ++kw) {
                float weight_grad_sum = 0.0f;
                
                for (int b = 0; b < batch_size; ++b) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            int ih = oh * stride_h_ - pad_h_ + kh;
                            int iw = ow * stride_w_ - pad_w_ + kw;
                            
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                weight_grad_sum += input.at(b, c, ih, iw) * output_grad.at(b, c, oh, ow);
                            }
                        }
                    }
                }
                
                weight_grad_.at(c, 0, kh, kw) = weight_grad_sum;
            }
        }
        
        // Compute bias gradient
        float bias_grad_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    bias_grad_sum += output_grad.at(b, c, oh, ow);
                }
            }
        }
        bias_grad_.at(c) = bias_grad_sum;
    }
}

void DepthwiseConv2DLayer::update_weights(float learning_rate) {
    float* weights_data = weights_.data();
    float* bias_data = bias_.data();
    float* weight_grad_data = weight_grad_.data();
    float* bias_grad_data = bias_grad_.data();
    
    for (int i = 0; i < weights_.size(); ++i) {
        weights_data[i] -= learning_rate * weight_grad_data[i];
    }
    
    for (int i = 0; i < bias_.size(); ++i) {
        bias_data[i] -= learning_rate * bias_grad_data[i];
    }
}

// ============================================================================
// ResidualBlock Implementation
// ============================================================================

ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride)
    : use_shortcut_(in_channels != out_channels || stride != 1) {
    
    // Main path
    conv1_ = std::make_unique<Conv2DLayer>(in_channels, out_channels, 3, 3, stride, stride, 1, 1);
    bn1_ = std::make_unique<BatchNormLayer>(out_channels);
    relu1_ = std::make_unique<ReLULayer>();
    
    conv2_ = std::make_unique<Conv2DLayer>(out_channels, out_channels, 3, 3, 1, 1, 1, 1);
    bn2_ = std::make_unique<BatchNormLayer>(out_channels);
    
    relu2_ = std::make_unique<ReLULayer>();
    
    // Shortcut path (if needed)
    if (use_shortcut_) {
        shortcut_conv_ = std::make_unique<Conv2DLayer>(in_channels, out_channels, 1, 1, stride, stride, 0, 0);
        shortcut_bn_ = std::make_unique<BatchNormLayer>(out_channels);
    }
}

Tensor ResidualBlock::forward(const Tensor& input, bool training) {
    last_input_ = input;
    
    // Main path
    Tensor x = conv1_->forward(input, training);
    x = bn1_->forward(x, training);
    x = relu1_->forward(x, training);
    
    x = conv2_->forward(x, training);
    x = bn2_->forward(x, training);
    
    // Shortcut path
    Tensor shortcut = input;
    if (use_shortcut_) {
        shortcut = shortcut_conv_->forward(input, training);
        shortcut = shortcut_bn_->forward(shortcut, training);
    }
    
    // Add shortcut
    float* x_data = x.data();
    const float* shortcut_data = shortcut.data();
    for (int i = 0; i < x.size(); ++i) {
        x_data[i] += shortcut_data[i];
    }
    
    // Final ReLU
    return relu2_->forward(x, training);
}

Tensor ResidualBlock::backward(const Tensor& output_gradient) {
    // Backward through final ReLU
    Tensor grad = relu2_->backward(output_gradient);
    
    // Split gradient for main path and shortcut
    Tensor main_grad = grad;
    Tensor shortcut_grad = grad;
    
    // Backward through main path
    main_grad = bn2_->backward(main_grad);
    main_grad = conv2_->backward(main_grad);
    main_grad = relu1_->backward(main_grad);
    main_grad = bn1_->backward(main_grad);
    main_grad = conv1_->backward(main_grad);
    
    // Backward through shortcut
    if (use_shortcut_) {
        shortcut_grad = shortcut_bn_->backward(shortcut_grad);
        shortcut_grad = shortcut_conv_->backward(shortcut_grad);
    }
    
    // Combine gradients
    float* main_data = main_grad.data();
    const float* shortcut_data = shortcut_grad.data();
    for (int i = 0; i < main_grad.size(); ++i) {
        main_data[i] += shortcut_data[i];
    }
    
    return main_grad;
}

void ResidualBlock::update_weights(float learning_rate) {
    conv1_->update_weights(learning_rate);
    bn1_->update_weights(learning_rate);
    conv2_->update_weights(learning_rate);
    bn2_->update_weights(learning_rate);
    
    if (use_shortcut_) {
        shortcut_conv_->update_weights(learning_rate);
        shortcut_bn_->update_weights(learning_rate);
    }
}

// ============================================================================
// InceptionModule Implementation
// ============================================================================

InceptionModule::InceptionModule(int in_channels, int branch1_channels, int branch2_channels, 
                               int branch3_channels, int branch4_channels)
    : in_channels_(in_channels), branch1_channels_(branch1_channels), 
      branch2_channels_(branch2_channels), branch3_channels_(branch3_channels),
      branch4_channels_(branch4_channels) {
    
    // Branch 1: 1x1 conv
    branch1_conv_ = std::make_unique<Conv2DLayer>(in_channels, branch1_channels, 1, 1, 1, 1, 0, 0);
    relu1_ = std::make_unique<ReLULayer>();
    
    // Branch 2: 1x1 -> 3x3 conv
    branch2_conv1_ = std::make_unique<Conv2DLayer>(in_channels, branch2_channels / 2, 1, 1, 1, 1, 0, 0);
    relu2a_ = std::make_unique<ReLULayer>();
    branch2_conv2_ = std::make_unique<Conv2DLayer>(branch2_channels / 2, branch2_channels, 3, 3, 1, 1, 1, 1);
    relu2b_ = std::make_unique<ReLULayer>();
    
    // Branch 3: 1x1 -> 5x5 conv  
    branch3_conv1_ = std::make_unique<Conv2DLayer>(in_channels, branch3_channels / 4, 1, 1, 1, 1, 0, 0);
    relu3a_ = std::make_unique<ReLULayer>();
    branch3_conv2_ = std::make_unique<Conv2DLayer>(branch3_channels / 4, branch3_channels, 5, 5, 1, 1, 2, 2);
    relu3b_ = std::make_unique<ReLULayer>();
    
    // Branch 4: maxpool -> 1x1 conv
    branch4_pool_ = std::make_unique<MaxPoolingLayer>(3, 3, 1, 1, 1, 1);
    branch4_conv_ = std::make_unique<Conv2DLayer>(in_channels, branch4_channels, 1, 1, 1, 1, 0, 0);
    relu4_ = std::make_unique<ReLULayer>();
}

Tensor InceptionModule::forward(const Tensor& input, bool training) {
    // Branch 1
    Tensor branch1_out = branch1_conv_->forward(input, training);
    branch1_out = relu1_->forward(branch1_out, training);
    
    // Branch 2
    Tensor branch2_out = branch2_conv1_->forward(input, training);
    branch2_out = relu2a_->forward(branch2_out, training);
    branch2_out = branch2_conv2_->forward(branch2_out, training);
    branch2_out = relu2b_->forward(branch2_out, training);
    
    // Branch 3
    Tensor branch3_out = branch3_conv1_->forward(input, training);
    branch3_out = relu3a_->forward(branch3_out, training);
    branch3_out = branch3_conv2_->forward(branch3_out, training);
    branch3_out = relu3b_->forward(branch3_out, training);
    
    // Branch 4
    Tensor branch4_out = branch4_pool_->forward(input, training);
    branch4_out = branch4_conv_->forward(branch4_out, training);
    branch4_out = relu4_->forward(branch4_out, training);
    
    // Concatenate all branches
    return concatenate_channels({branch1_out, branch2_out, branch3_out, branch4_out});
}

Tensor InceptionModule::concatenate_channels(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    auto first_shape = tensors[0].shape();
    if (first_shape.size() != 4) {
        throw std::runtime_error("Expected 4D tensors for channel concatenation");
    }
    
    int batch_size = first_shape[0];
    int total_channels = 0;
    int height = first_shape[2];
    int width = first_shape[3];
    
    // Calculate total channels
    for (const auto& tensor : tensors) {
        auto shape = tensor.shape();
        total_channels += shape[1];
        
        // Verify dimensions match
        if (shape[0] != batch_size || shape[2] != height || shape[3] != width) {
            throw std::runtime_error("Tensor dimensions must match except for channels");
        }
    }
    
    Tensor result({batch_size, total_channels, height, width});
    
    // Copy data
    int channel_offset = 0;
    for (const auto& tensor : tensors) {
        auto shape = tensor.shape();
        int tensor_channels = shape[1];
        
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < tensor_channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        result.at(b, channel_offset + c, h, w) = tensor.at(b, c, h, w);
                    }
                }
            }
        }
        channel_offset += tensor_channels;
    }
    
    return result;
}

std::vector<Tensor> InceptionModule::split_channels(const Tensor& tensor, const std::vector<int>& channel_sizes) {
    auto shape = tensor.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D tensor for channel splitting");
    }
    
    int batch_size = shape[0];
    int height = shape[2];
    int width = shape[3];
    
    std::vector<Tensor> result;
    int channel_offset = 0;
    
    for (int num_channels : channel_sizes) {
        Tensor split_tensor({batch_size, num_channels, height, width});
        
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        split_tensor.at(b, c, h, w) = tensor.at(b, channel_offset + c, h, w);
                    }
                }
            }
        }
        
        result.push_back(split_tensor);
        channel_offset += num_channels;
    }
    
    return result;
}

Tensor InceptionModule::backward(const Tensor& output_gradient) {
    // Split gradient by channels
    std::vector<int> channel_sizes = {branch1_channels_, branch2_channels_, branch3_channels_, branch4_channels_};
    auto split_grads = split_channels(output_gradient, channel_sizes);
    
    // Backward through each branch
    Tensor branch1_grad = relu1_->backward(split_grads[0]);
    branch1_grad = branch1_conv_->backward(branch1_grad);
    
    Tensor branch2_grad = relu2b_->backward(split_grads[1]);
    branch2_grad = branch2_conv2_->backward(branch2_grad);
    branch2_grad = relu2a_->backward(branch2_grad);
    branch2_grad = branch2_conv1_->backward(branch2_grad);
    
    Tensor branch3_grad = relu3b_->backward(split_grads[2]);
    branch3_grad = branch3_conv2_->backward(branch3_grad);
    branch3_grad = relu3a_->backward(branch3_grad);
    branch3_grad = branch3_conv1_->backward(branch3_grad);
    
    Tensor branch4_grad = relu4_->backward(split_grads[3]);
    branch4_grad = branch4_conv_->backward(branch4_grad);
    branch4_grad = branch4_pool_->backward(branch4_grad);
    
    // Sum all input gradients
    Tensor input_grad = branch1_grad;
    float* input_grad_data = input_grad.data();
    const float* branch2_data = branch2_grad.data();
    const float* branch3_data = branch3_grad.data();
    const float* branch4_data = branch4_grad.data();
    
    for (int i = 0; i < input_grad.size(); ++i) {
        input_grad_data[i] += branch2_data[i] + branch3_data[i] + branch4_data[i];
    }
    
    return input_grad;
}

void InceptionModule::update_weights(float learning_rate) {
    branch1_conv_->update_weights(learning_rate);
    branch2_conv1_->update_weights(learning_rate);
    branch2_conv2_->update_weights(learning_rate);
    branch3_conv1_->update_weights(learning_rate);
    branch3_conv2_->update_weights(learning_rate);
    branch4_conv_->update_weights(learning_rate);
}

// ============================================================================
// BottleneckBlock Implementation
// ============================================================================

BottleneckBlock::BottleneckBlock(int in_channels, int bottleneck_channels, int out_channels, int stride)
    : use_shortcut_(in_channels != out_channels || stride != 1) {
    
    // Main path
    conv1_ = std::make_unique<Conv2DLayer>(in_channels, bottleneck_channels, 1, 1, 1, 1, 0, 0);
    bn1_ = std::make_unique<BatchNormLayer>(bottleneck_channels);
    relu1_ = std::make_unique<ReLULayer>();
    
    conv2_ = std::make_unique<Conv2DLayer>(bottleneck_channels, bottleneck_channels, 3, 3, stride, stride, 1, 1);
    bn2_ = std::make_unique<BatchNormLayer>(bottleneck_channels);
    relu2_ = std::make_unique<ReLULayer>();
    
    conv3_ = std::make_unique<Conv2DLayer>(bottleneck_channels, out_channels, 1, 1, 1, 1, 0, 0);
    bn3_ = std::make_unique<BatchNormLayer>(out_channels);
    
    final_relu_ = std::make_unique<ReLULayer>();
    
    // Shortcut path
    if (use_shortcut_) {
        shortcut_conv_ = std::make_unique<Conv2DLayer>(in_channels, out_channels, 1, 1, stride, stride, 0, 0);
        shortcut_bn_ = std::make_unique<BatchNormLayer>(out_channels);
    }
}

Tensor BottleneckBlock::forward(const Tensor& input, bool training) {
    last_input_ = input;
    
    // Main path
    Tensor x = conv1_->forward(input, training);
    x = bn1_->forward(x, training);
    x = relu1_->forward(x, training);
    
    x = conv2_->forward(x, training);
    x = bn2_->forward(x, training);
    x = relu2_->forward(x, training);
    
    x = conv3_->forward(x, training);
    x = bn3_->forward(x, training);
    
    // Shortcut path
    Tensor shortcut = input;
    if (use_shortcut_) {
        shortcut = shortcut_conv_->forward(input, training);
        shortcut = shortcut_bn_->forward(shortcut, training);
    }
    
    // Add shortcut
    float* x_data = x.data();
    const float* shortcut_data = shortcut.data();
    for (int i = 0; i < x.size(); ++i) {
        x_data[i] += shortcut_data[i];
    }
    
    // Final ReLU
    return final_relu_->forward(x, training);
}

Tensor BottleneckBlock::backward(const Tensor& output_gradient) {
    // Backward through final ReLU
    Tensor grad = final_relu_->backward(output_gradient);
    
    // Split gradient for main path and shortcut
    Tensor main_grad = grad;
    Tensor shortcut_grad = grad;
    
    // Backward through main path
    main_grad = bn3_->backward(main_grad);
    main_grad = conv3_->backward(main_grad);
    main_grad = relu2_->backward(main_grad);
    main_grad = bn2_->backward(main_grad);
    main_grad = conv2_->backward(main_grad);
    main_grad = relu1_->backward(main_grad);
    main_grad = bn1_->backward(main_grad);
    main_grad = conv1_->backward(main_grad);
    
    // Backward through shortcut
    if (use_shortcut_) {
        shortcut_grad = shortcut_bn_->backward(shortcut_grad);
        shortcut_grad = shortcut_conv_->backward(shortcut_grad);
    }
    
    // Combine gradients
    float* main_data = main_grad.data();
    const float* shortcut_data = shortcut_grad.data();
    for (int i = 0; i < main_grad.size(); ++i) {
        main_data[i] += shortcut_data[i];
    }
    
    return main_grad;
}

void BottleneckBlock::update_weights(float learning_rate) {
    conv1_->update_weights(learning_rate);
    bn1_->update_weights(learning_rate);
    conv2_->update_weights(learning_rate);
    bn2_->update_weights(learning_rate);
    conv3_->update_weights(learning_rate);
    bn3_->update_weights(learning_rate);
    
    if (use_shortcut_) {
        shortcut_conv_->update_weights(learning_rate);
        shortcut_bn_->update_weights(learning_rate);
    }
}

} // namespace cnn
