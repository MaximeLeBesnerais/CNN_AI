#include "cnn/layers.h"
#include <stdexcept>
#include <cmath>

namespace cnn {

// Flatten Layer Implementation
Tensor FlattenLayer::forward(const Tensor& input, bool training) {
    original_shape_ = input.shape();
    return input.flatten();
}

Tensor FlattenLayer::backward(const Tensor& output_gradient) {
    if (original_shape_.empty()) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    return output_gradient.reshape(original_shape_);
}

// BatchNorm Layer Implementation
BatchNormLayer::BatchNormLayer(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      gamma_({num_features}), beta_({num_features}),
      gamma_grad_({num_features}), beta_grad_({num_features}),
      running_mean_({num_features}), running_var_({num_features}) {
    
    // Initialize parameters
    gamma_.fill(1.0f);  // Scale starts at 1
    beta_.zero();       // Shift starts at 0
    
    // Initialize running statistics
    running_mean_.zero();
    running_var_.fill(1.0f);
}

Tensor BatchNormLayer::forward(const Tensor& input, bool training) {
    // Input can be 2D (N, C) or 4D (N, C, H, W)
    if (input.ndim() != 2 && input.ndim() != 4) {
        throw std::invalid_argument("BatchNorm input must be 2D or 4D");
    }
    
    int batch_size = input.shape()[0];
    int channels = input.shape()[1];
    
    if (channels != num_features_) {
        throw std::invalid_argument("Input channels don't match BatchNorm features");
    }
    
    cached_input_ = input;
    
    if (training) {
        // Compute batch statistics
        if (input.ndim() == 2) {
            // 2D case: (N, C)
            cached_mean_ = input.sum(0) / static_cast<float>(batch_size);
            
            // Compute variance
            Tensor centered = input - cached_mean_;
            cached_var_ = (centered * centered).sum(0) / static_cast<float>(batch_size);
        } else {
            // 4D case: (N, C, H, W) - need to compute mean/var over N, H, W dimensions
            int spatial_size = input.shape()[2] * input.shape()[3];
            int total_elements = batch_size * spatial_size;
            
            cached_mean_ = Tensor({channels});
            cached_var_ = Tensor({channels});
            
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int n = 0; n < batch_size; ++n) {
                    for (int h = 0; h < input.shape()[2]; ++h) {
                        for (int w = 0; w < input.shape()[3]; ++w) {
                            sum += input.at(n, c, h, w);
                        }
                    }
                }
                cached_mean_.at(c) = sum / total_elements;
                
                float var_sum = 0.0f;
                for (int n = 0; n < batch_size; ++n) {
                    for (int h = 0; h < input.shape()[2]; ++h) {
                        for (int w = 0; w < input.shape()[3]; ++w) {
                            float diff = input.at(n, c, h, w) - cached_mean_.at(c);
                            var_sum += diff * diff;
                        }
                    }
                }
                cached_var_.at(c) = var_sum / total_elements;
            }
        }
        
        // Update running statistics
        for (int c = 0; c < num_features_; ++c) {
            running_mean_.at(c) = (1.0f - momentum_) * running_mean_.at(c) + 
                                  momentum_ * cached_mean_.at(c);
            running_var_.at(c) = (1.0f - momentum_) * running_var_.at(c) + 
                                 momentum_ * cached_var_.at(c);
        }
    } else {
        // Use running statistics for inference
        cached_mean_ = running_mean_;
        cached_var_ = running_var_;
    }
    
    // Normalize
    Tensor output(input.shape());
    
    if (input.ndim() == 2) {
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                float normalized = (input.at(n, c) - cached_mean_.at(c)) / 
                                  std::sqrt(cached_var_.at(c) + eps_);
                output.at(n, c) = gamma_.at(c) * normalized + beta_.at(c);
            }
        }
    } else {
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < input.shape()[2]; ++h) {
                    for (int w = 0; w < input.shape()[3]; ++w) {
                        float normalized = (input.at(n, c, h, w) - cached_mean_.at(c)) / 
                                          std::sqrt(cached_var_.at(c) + eps_);
                        output.at(n, c, h, w) = gamma_.at(c) * normalized + beta_.at(c);
                    }
                }
            }
        }
    }
    
    // Cache normalized values for backward pass
    cached_normalized_ = Tensor(input.shape());
    for (int i = 0; i < input.size(); ++i) {
        int c = (input.ndim() == 2) ? i % channels : 
                (i / (input.shape()[2] * input.shape()[3])) % channels;
        cached_normalized_.data()[i] = (input.data()[i] - cached_mean_.at(c)) / 
                                       std::sqrt(cached_var_.at(c) + eps_);
    }
    
    return output;
}

Tensor BatchNormLayer::backward(const Tensor& output_gradient) {
    if (cached_input_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    int batch_size = cached_input_.shape()[0];
    int channels = cached_input_.shape()[1];
    
    // Compute gradients for gamma and beta
    beta_grad_.zero();
    gamma_grad_.zero();
    
    if (cached_input_.ndim() == 2) {
        // 2D case
        for (int c = 0; c < channels; ++c) {
            for (int n = 0; n < batch_size; ++n) {
                beta_grad_.at(c) += output_gradient.at(n, c);
                gamma_grad_.at(c) += output_gradient.at(n, c) * cached_normalized_.at(n, c);
            }
        }
    } else {
        // 4D case
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < cached_input_.shape()[2]; ++h) {
                    for (int w = 0; w < cached_input_.shape()[3]; ++w) {
                        beta_grad_.at(c) += output_gradient.at(n, c, h, w);
                        gamma_grad_.at(c) += output_gradient.at(n, c, h, w) * 
                                            cached_normalized_.at(n, c, h, w);
                    }
                }
            }
        }
    }
    
    // Compute input gradient (simplified implementation)
    Tensor input_gradient(cached_input_.shape());
    
    int total_elements = (cached_input_.ndim() == 2) ? batch_size : 
                        batch_size * cached_input_.shape()[2] * cached_input_.shape()[3];
    
    for (int i = 0; i < cached_input_.size(); ++i) {
        int c = (cached_input_.ndim() == 2) ? i % channels : 
                (i / (cached_input_.shape()[2] * cached_input_.shape()[3])) % channels;
        
        float std_inv = 1.0f / std::sqrt(cached_var_.at(c) + eps_);
        input_gradient.data()[i] = gamma_.at(c) * std_inv * output_gradient.data()[i];
    }
    
    return input_gradient;
}

void BatchNormLayer::update_weights(float learning_rate) {
    gamma_ -= gamma_grad_ * learning_rate;
    beta_ -= beta_grad_ * learning_rate;
}

std::vector<Tensor*> BatchNormLayer::get_parameters() {
    return {&gamma_, &beta_};
}

std::vector<Tensor*> BatchNormLayer::get_gradients() {
    return {&gamma_grad_, &beta_grad_};
}

} // namespace cnn
