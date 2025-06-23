#include "cnn/conv2d.h"
#include <stdexcept>
#include <cmath>

namespace cnn {

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, 
                         int kernel_size, int stride, int padding, bool use_bias)
    : Conv2DLayer(in_channels, out_channels, kernel_size, kernel_size, 
                  stride, stride, padding, padding, use_bias) {}

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels,
                         int kernel_height, int kernel_width,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_height_(kernel_height), kernel_width_(kernel_width),
      stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w), use_bias_(use_bias) {
    
    init_parameters();
}

void Conv2DLayer::init_parameters() {
    // Initialize weights: (out_channels, in_channels, kernel_h, kernel_w)
    weights_ = Tensor({out_channels_, in_channels_, kernel_height_, kernel_width_});
    weight_grad_ = Tensor({out_channels_, in_channels_, kernel_height_, kernel_width_});
    
    if (use_bias_) {
        bias_ = Tensor({out_channels_});
        bias_grad_ = Tensor({out_channels_});
        bias_.zero();
    }
    
    // Default initialization
    xavier_init();
}

std::vector<int> Conv2DLayer::compute_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Input must be 4D: (N, C, H, W)");
    }
    
    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];
    
    if (C != in_channels_) {
        throw std::invalid_argument("Input channels don't match layer configuration");
    }
    
    int out_h = (H + 2 * pad_h_ - kernel_height_) / stride_h_ + 1;
    int out_w = (W + 2 * pad_w_ - kernel_width_) / stride_w_ + 1;
    
    return {N, out_channels_, out_h, out_w};
}

Tensor Conv2DLayer::forward(const Tensor& input, bool training) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Conv2D input must be 4D: (N, C, H, W)");
    }
    
    // Cache input for backward pass
    cached_input_ = input;
    output_shape_ = compute_output_shape(input.shape());
    
    int N = input.shape()[0];
    int out_h = output_shape_[2];
    int out_w = output_shape_[3];
    
    // Convert input to column matrix using im2col
    cached_im2col_ = input.im2col(kernel_height_, kernel_width_, 
                                  stride_h_, stride_w_, pad_h_, pad_w_);
    
    // Reshape weights for matrix multiplication
    // weights: (out_channels, in_channels * kernel_h * kernel_w)
    auto weight_matrix = weights_.reshape({out_channels_, 
                                          in_channels_ * kernel_height_ * kernel_width_});
    
    // Perform convolution as matrix multiplication
    // im2col: (N * out_h * out_w, in_channels * kernel_h * kernel_w)
    // weight_matrix: (out_channels, in_channels * kernel_h * kernel_w)
    // result: (N * out_h * out_w, out_channels)
    auto conv_result = cached_im2col_.matmul(weight_matrix.transpose());
    
    // Add bias if enabled
    if (use_bias_) {
        // Broadcast bias across all spatial locations
        for (int i = 0; i < conv_result.shape()[0]; ++i) {
            for (int j = 0; j < out_channels_; ++j) {
                conv_result.at(i, j) += bias_.at(j);
            }
        }
    }
    
    // Reshape result back to (N, out_channels, out_h, out_w)
    auto output = conv_result.reshape({N, out_h, out_w, out_channels_});
    // Transpose to correct format: (N, out_channels, out_h, out_w)
    output = output.transpose(1, 3).transpose(2, 3);
    
    return output;
}

Tensor Conv2DLayer::backward(const Tensor& output_gradient) {
    if (cached_input_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    int N = cached_input_.shape()[0];
    int out_h = output_shape_[2];
    int out_w = output_shape_[3];
    
    // Reshape output gradient for matrix operations
    // output_gradient: (N, out_channels, out_h, out_w)
    // Transpose and reshape to: (N * out_h * out_w, out_channels)
    auto grad_reshaped = output_gradient.transpose(1, 2).transpose(2, 3);
    grad_reshaped = grad_reshaped.reshape({N * out_h * out_w, out_channels_});
    
    // Compute bias gradient (sum over batch and spatial dimensions)
    if (use_bias_) {
        bias_grad_ = grad_reshaped.sum(0);
    }
    
    // Compute weight gradient
    // weight_grad = im2col^T @ grad_reshaped
    // cached_im2col_: (N * out_h * out_w, in_channels * kernel_h * kernel_w)
    // grad_reshaped: (N * out_h * out_w, out_channels)
    // result: (in_channels * kernel_h * kernel_w, out_channels)
    auto weight_grad_flat = cached_im2col_.transpose().matmul(grad_reshaped);
    
    // Reshape back to weight shape
    weight_grad_ = weight_grad_flat.transpose().reshape(weights_.shape());
    
    // Compute input gradient
    // Reshape weights for backward convolution
    auto weight_matrix = weights_.reshape({out_channels_, 
                                          in_channels_ * kernel_height_ * kernel_width_});
    
    // input_grad_im2col = grad_reshaped @ weight_matrix
    auto input_grad_im2col = grad_reshaped.matmul(weight_matrix);
    
    // Convert back from im2col format to original input format using col2im
    auto input_gradient = Tensor::col2im(input_grad_im2col, cached_input_.shape(),
                                        kernel_height_, kernel_width_,
                                        stride_h_, stride_w_, pad_h_, pad_w_);
    
    return input_gradient;
}

void Conv2DLayer::update_weights(float learning_rate) {
    weights_ -= weight_grad_ * learning_rate;
    if (use_bias_) {
        bias_ -= bias_grad_ * learning_rate;
    }
}

std::vector<Tensor*> Conv2DLayer::get_parameters() {
    if (use_bias_) {
        return {&weights_, &bias_};
    } else {
        return {&weights_};
    }
}

std::vector<Tensor*> Conv2DLayer::get_gradients() {
    if (use_bias_) {
        return {&weight_grad_, &bias_grad_};
    } else {
        return {&weight_grad_};
    }
}

void Conv2DLayer::xavier_init() {
    int fan_in = in_channels_ * kernel_height_ * kernel_width_;
    int fan_out = out_channels_ * kernel_height_ * kernel_width_;
    weights_.xavier_init(fan_in, fan_out);
}

void Conv2DLayer::he_init() {
    int fan_in = in_channels_ * kernel_height_ * kernel_width_;
    weights_.he_init(fan_in);
}

void Conv2DLayer::random_init(float std) {
    weights_.random_normal(0.0f, std);
}

} // namespace cnn
