#include "cnn/activation.h"
#include <cmath>
#include <algorithm>

namespace cnn {

// ReLU Layer
Tensor ReLULayer::forward(const Tensor& input, bool training) {
    cached_input_ = input;
    return input.relu();
}

Tensor ReLULayer::backward(const Tensor& output_gradient) {
    if (cached_input_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    // Element-wise multiplication with ReLU derivative
    return output_gradient * cached_input_.relu_derivative();
}

// Leaky ReLU Layer
Tensor LeakyReLULayer::forward(const Tensor& input, bool training) {
    cached_input_ = input;
    return input.leaky_relu(alpha_);
}

Tensor LeakyReLULayer::backward(const Tensor& output_gradient) {
    if (cached_input_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    return output_gradient * cached_input_.leaky_relu_derivative(alpha_);
}

// Sigmoid Layer
Tensor SigmoidLayer::forward(const Tensor& input, bool training) {
    cached_output_ = input.sigmoid();
    return cached_output_;
}

Tensor SigmoidLayer::backward(const Tensor& output_gradient) {
    if (cached_output_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    Tensor ones = Tensor::ones(cached_output_.shape());
    Tensor derivative = cached_output_ * (ones - cached_output_);
    return output_gradient * derivative;
}

// Tanh Layer
Tensor TanhLayer::forward(const Tensor& input, bool training) {
    cached_output_ = input.tanh();
    return cached_output_;
}

Tensor TanhLayer::backward(const Tensor& output_gradient) {
    if (cached_output_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    // Tanh derivative: 1 - tanh^2(x)
    Tensor ones = Tensor::ones(cached_output_.shape());
    Tensor derivative = ones - cached_output_ * cached_output_;
    return output_gradient * derivative;
}

// Softmax Layer
Tensor SoftmaxLayer::forward(const Tensor& input, bool training) {
    cached_output_ = input.softmax(axis_);
    return cached_output_;
}

Tensor SoftmaxLayer::backward(const Tensor& output_gradient) {
    if (cached_output_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    // Softmax gradient is more complex
    // For simplicity, implementing the common case where softmax is used with cross-entropy
    // In that case, the combined gradient is just (output - target), which is handled in the loss function
    // For general case, we need the Jacobian matrix
    
    // Simplified implementation for common use case
    return output_gradient;
}

} // namespace cnn
