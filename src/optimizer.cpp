#include "cnn/optimizer.h"
#include <stdexcept>
#include <cmath>

namespace cnn {

// SGD Optimizer Implementation
SGDOptimizer::SGDOptimizer(const std::vector<Tensor*>& parameters,
                           const std::vector<Tensor*>& gradients,
                           float learning_rate,
                           float momentum)
    : Optimizer(parameters, gradients, learning_rate), momentum_(momentum) {
    
    // Initialize velocity for momentum
    if (momentum_ > 0.0f) {
        velocity_.reserve(parameters_.size());
        for (const auto& param : parameters_) {
            velocity_.emplace_back(param->shape());
            velocity_.back().zero();
        }
    }
}

void SGDOptimizer::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (momentum_ > 0.0f) {
            // SGD with momentum: v = momentum * v + lr * grad
            velocity_[i] *= momentum_;
            velocity_[i] += (*gradients_[i]) * learning_rate_;
            *parameters_[i] -= velocity_[i];
        } else {
            // Simple SGD: param = param - lr * grad
            *parameters_[i] -= (*gradients_[i]) * learning_rate_;
        }
    }
}

void SGDOptimizer::zero_grad() {
    for (auto& grad : gradients_) {
        grad->zero();
    }
}

// Adam Optimizer Implementation
AdamOptimizer::AdamOptimizer(const std::vector<Tensor*>& parameters,
                             const std::vector<Tensor*>& gradients,
                             float learning_rate,
                             float beta1, float beta2, float epsilon)
    : Optimizer(parameters, gradients, learning_rate),
      beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    
    // Initialize first and second moment estimates
    m_.reserve(parameters_.size());
    v_.reserve(parameters_.size());
    
    for (const auto& param : parameters_) {
        m_.emplace_back(param->shape());
        v_.emplace_back(param->shape());
        m_.back().zero();
        v_.back().zero();
    }
}

void AdamOptimizer::step() {
    t_++;  // Increment time step
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        m_[i] *= beta1_;
        m_[i] += (*gradients_[i]) * (1.0f - beta1_);
        
        // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
        v_[i] *= beta2_;
        v_[i] += ((*gradients_[i]) * (*gradients_[i])) * (1.0f - beta2_);
        
        // Compute bias-corrected first moment estimate
        float bias_correction1 = 1.0f - std::pow(beta1_, t_);
        float bias_correction2 = 1.0f - std::pow(beta2_, t_);
        
        // Update parameters
        for (int j = 0; j < parameters_[i]->size(); ++j) {
            float m_hat = m_[i].data()[j] / bias_correction1;
            float v_hat = v_[i].data()[j] / bias_correction2;
            
            parameters_[i]->data()[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void AdamOptimizer::zero_grad() {
    for (auto& grad : gradients_) {
        grad->zero();
    }
}

// RMSProp Optimizer Implementation
RMSPropOptimizer::RMSPropOptimizer(const std::vector<Tensor*>& parameters,
                                   const std::vector<Tensor*>& gradients,
                                   float learning_rate,
                                   float alpha, float epsilon)
    : Optimizer(parameters, gradients, learning_rate),
      alpha_(alpha), epsilon_(epsilon) {
    
    // Initialize squared gradient average
    squared_avg_.reserve(parameters_.size());
    for (const auto& param : parameters_) {
        squared_avg_.emplace_back(param->shape());
        squared_avg_.back().zero();
    }
}

void RMSPropOptimizer::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        // Update squared gradient average: avg = alpha * avg + (1 - alpha) * grad^2
        squared_avg_[i] *= alpha_;
        squared_avg_[i] += ((*gradients_[i]) * (*gradients_[i])) * (1.0f - alpha_);
        
        // Update parameters: param = param - lr * grad / (sqrt(avg) + eps)
        for (int j = 0; j < parameters_[i]->size(); ++j) {
            float rms = std::sqrt(squared_avg_[i].data()[j] + epsilon_);
            parameters_[i]->data()[j] -= learning_rate_ * gradients_[i]->data()[j] / rms;
        }
    }
}

void RMSPropOptimizer::zero_grad() {
    for (auto& grad : gradients_) {
        grad->zero();
    }
}

} // namespace cnn
