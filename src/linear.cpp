#include "cnn/linear.h"
#include <stdexcept>

namespace cnn {

LinearLayer::LinearLayer(int input_size, int output_size) 
    : input_size_(input_size), output_size_(output_size),
      weights_({input_size, output_size}),
      bias_({output_size}),
      weight_grad_({input_size, output_size}),
      bias_grad_({output_size}) {
    
    // Initialize with Xavier initialization by default
    xavier_init();
}

Tensor LinearLayer::forward(const Tensor& input, bool training) {
    // Input shape: (batch_size, input_size) or (input_size,)
    // Weights shape: (input_size, output_size)
    // Output shape: (batch_size, output_size) or (output_size,)
    
    if (input.shape().back() != input_size_) {
        throw std::invalid_argument("Input size doesn't match layer input size");
    }
    
    // Cache input for backward pass
    cached_input_ = input;
    
    Tensor output;
    if (input.ndim() == 1) {
        // Single sample: input shape (input_size,)
        output = input.matmul(weights_);
        // Add bias element-wise
        for (int i = 0; i < output_size_; ++i) {
            output.at(i) += bias_.at(i);
        }
    } else if (input.ndim() == 2) {
        // Batch: input shape (batch_size, input_size)
        output = input.matmul(weights_);
        // Add bias to each sample in the batch
        int batch_size = input.shape()[0];
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < output_size_; ++i) {
                output.at(b, i) += bias_.at(i);
            }
        }
    } else {
        throw std::invalid_argument("Input must be 1D or 2D tensor");
    }
    
    return output;
}

Tensor LinearLayer::backward(const Tensor& output_gradient) {
    // output_gradient shape: same as forward output
    // Need to compute gradients for weights, bias, and input
    
    if (cached_input_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    // Gradient w.r.t. bias: sum over batch dimension
    if (output_gradient.ndim() == 1) {
        bias_grad_ = output_gradient;
    } else {
        bias_grad_ = output_gradient.sum(0);
    }
    
    // Gradient w.r.t. weights: input^T @ output_gradient
    if (cached_input_.ndim() == 1) {
        // Single sample case
        auto input_col = cached_input_.reshape({cached_input_.size(), 1});
        auto grad_row = output_gradient.reshape({1, output_gradient.size()});
        weight_grad_ = input_col.matmul(grad_row);
    } else {
        // Batch case
        weight_grad_ = cached_input_.transpose().matmul(output_gradient);
    }
    
    // Gradient w.r.t. input: output_gradient @ weights^T
    Tensor input_gradient = output_gradient.matmul(weights_.transpose());
    
    return input_gradient;
}

void LinearLayer::update_weights(float learning_rate) {
    weights_ -= weight_grad_ * learning_rate;
    bias_ -= bias_grad_ * learning_rate;
}

std::vector<Tensor*> LinearLayer::get_parameters() {
    return {&weights_, &bias_};
}

std::vector<Tensor*> LinearLayer::get_gradients() {
    return {&weight_grad_, &bias_grad_};
}

void LinearLayer::xavier_init() {
    weights_.xavier_init(input_size_, output_size_);
    bias_.zero();
}

void LinearLayer::he_init() {
    weights_.he_init(input_size_);
    bias_.zero();
}

void LinearLayer::zero_init() {
    weights_.zero();
    bias_.zero();
}

void LinearLayer::random_init(float std) {
    weights_.random_normal(0.0f, std);
    bias_.zero();
}

} // namespace cnn
