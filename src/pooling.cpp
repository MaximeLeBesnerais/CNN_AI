#include "cnn/pooling.h"
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace cnn {

PoolingLayer::PoolingLayer(PoolingType pool_type, int kernel_size, 
                          int stride, int padding)
    : PoolingLayer(pool_type, kernel_size, kernel_size, 
                   stride == -1 ? kernel_size : stride, 
                   stride == -1 ? kernel_size : stride, 
                   padding, padding) {}

PoolingLayer::PoolingLayer(PoolingType pool_type, int kernel_h, int kernel_w,
                          int stride_h, int stride_w, int pad_h, int pad_w)
    : pool_type_(pool_type), kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {}

std::vector<int> PoolingLayer::compute_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Pooling input must be 4D: (N, C, H, W)");
    }
    
    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];
    
    int out_h = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    int out_w = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
    
    return {N, C, out_h, out_w};
}

Tensor PoolingLayer::forward(const Tensor& input, bool training) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Pooling input must be 4D: (N, C, H, W)");
    }
    
    cached_input_ = input;
    output_shape_ = compute_output_shape(input.shape());
    
    if (pool_type_ == PoolingType::MAX) {
        return max_pool_forward(input);
    } else {
        return avg_pool_forward(input);
    }
}

Tensor PoolingLayer::backward(const Tensor& output_gradient) {
    if (cached_input_.size() == 0) {
        throw std::runtime_error("Must call forward before backward");
    }
    
    if (pool_type_ == PoolingType::MAX) {
        return max_pool_backward(output_gradient);
    } else {
        return avg_pool_backward(output_gradient);
    }
}

std::string PoolingLayer::name() const {
    return pool_type_ == PoolingType::MAX ? "MaxPooling" : "AvgPooling";
}

Tensor PoolingLayer::max_pool_forward(const Tensor& input) {
    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];
    
    int out_h = output_shape_[2];
    int out_w = output_shape_[3];
    
    Tensor output(output_shape_);
    max_indices_ = Tensor({N, C, out_h, out_w});  // Store flat indices for backward pass
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;
                    
                    for (int kh = 0; kh < kernel_h_; ++kh) {
                        for (int kw = 0; kw < kernel_w_; ++kw) {
                            int ih = oh * stride_h_ + kh - pad_h_;
                            int iw = ow * stride_w_ + kw - pad_w_;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float val = input.at(n, c, ih, iw);
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = ih * W + iw;  // Flat index in spatial dimensions
                                }
                            }
                        }
                    }
                    
                    output.at(n, c, oh, ow) = max_val;
                    max_indices_.at(n, c, oh, ow) = static_cast<float>(max_idx);
                }
            }
        }
    }
    
    return output;
}

Tensor PoolingLayer::avg_pool_forward(const Tensor& input) {
    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];
    
    int out_h = output_shape_[2];
    int out_w = output_shape_[3];
    
    Tensor output(output_shape_);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (int kh = 0; kh < kernel_h_; ++kh) {
                        for (int kw = 0; kw < kernel_w_; ++kw) {
                            int ih = oh * stride_h_ + kh - pad_h_;
                            int iw = ow * stride_w_ + kw - pad_w_;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input.at(n, c, ih, iw);
                                count++;
                            }
                        }
                    }
                    
                    output.at(n, c, oh, ow) = count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }
    
    return output;
}

Tensor PoolingLayer::max_pool_backward(const Tensor& output_gradient) {
    int N = cached_input_.shape()[0];
    int C = cached_input_.shape()[1];
    int H = cached_input_.shape()[2];
    int W = cached_input_.shape()[3];
    
    int out_h = output_shape_[2];
    int out_w = output_shape_[3];
    
    Tensor input_gradient(cached_input_.shape());
    input_gradient.zero();
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int max_idx = static_cast<int>(max_indices_.at(n, c, oh, ow));
                    if (max_idx >= 0) {
                        int ih = max_idx / W;
                        int iw = max_idx % W;
                        input_gradient.at(n, c, ih, iw) += output_gradient.at(n, c, oh, ow);
                    }
                }
            }
        }
    }
    
    return input_gradient;
}

Tensor PoolingLayer::avg_pool_backward(const Tensor& output_gradient) {
    int N = cached_input_.shape()[0];
    int C = cached_input_.shape()[1];
    int H = cached_input_.shape()[2];
    int W = cached_input_.shape()[3];
    
    int out_h = output_shape_[2];
    int out_w = output_shape_[3];
    
    Tensor input_gradient(cached_input_.shape());
    input_gradient.zero();
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float grad_val = output_gradient.at(n, c, oh, ow);
                    int count = 0;
                    
                    // Count valid positions in kernel
                    for (int kh = 0; kh < kernel_h_; ++kh) {
                        for (int kw = 0; kw < kernel_w_; ++kw) {
                            int ih = oh * stride_h_ + kh - pad_h_;
                            int iw = ow * stride_w_ + kw - pad_w_;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                count++;
                            }
                        }
                    }
                    
                    // Distribute gradient equally
                    float distributed_grad = count > 0 ? grad_val / count : 0.0f;
                    
                    for (int kh = 0; kh < kernel_h_; ++kh) {
                        for (int kw = 0; kw < kernel_w_; ++kw) {
                            int ih = oh * stride_h_ + kh - pad_h_;
                            int iw = ow * stride_w_ + kw - pad_w_;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                input_gradient.at(n, c, ih, iw) += distributed_grad;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return input_gradient;
}

} // namespace cnn
