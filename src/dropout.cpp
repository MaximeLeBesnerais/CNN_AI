#include "cnn/dropout.h"
#include <stdexcept>

namespace cnn {

DropoutLayer::DropoutLayer(float dropout_rate) 
    : dropout_rate_(dropout_rate), generator_(std::random_device{}()), distribution_(0.0f, 1.0f) {
    
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
}

Tensor DropoutLayer::forward(const Tensor& input, bool training) {
    if (!training) {
        // During inference, return input as-is
        return input;
    }
    
    // During training, apply dropout
    mask_ = Tensor(input.shape());
    float keep_prob = 1.0f - dropout_rate_;
    
    // Generate random mask
    for (int i = 0; i < mask_.size(); ++i) {
        mask_.data()[i] = (distribution_(generator_) < keep_prob) ? (1.0f / keep_prob) : 0.0f;
    }
    
    // Apply mask
    return input * mask_;
}

Tensor DropoutLayer::backward(const Tensor& output_gradient) {
    if (mask_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    // Apply the same mask to the gradient
    return output_gradient * mask_;
}

void DropoutLayer::set_dropout_rate(float rate) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
    dropout_rate_ = rate;
}

} // namespace cnn
