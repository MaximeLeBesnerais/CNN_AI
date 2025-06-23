#pragma once

#include "cnn/layer.h"

namespace cnn {

enum class PoolingType {
    MAX,
    AVERAGE
};

class PoolingLayer : public Layer {
private:
    PoolingType pool_type_;
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    
    // For max pooling backward pass
    Tensor max_indices_;
    Tensor cached_input_;
    std::vector<int> output_shape_;
    
public:
    PoolingLayer(PoolingType pool_type, int kernel_size, 
                 int stride = -1, int padding = 0);
    
    PoolingLayer(PoolingType pool_type, int kernel_h, int kernel_w,
                 int stride_h, int stride_w, 
                 int pad_h, int pad_w);
    
    Tensor forward(const Tensor& input, bool training = true) override;
    Tensor backward(const Tensor& output_gradient) override;
    std::string name() const override;
    
private:
    std::vector<int> compute_output_shape(const std::vector<int>& input_shape) const;
    Tensor max_pool_forward(const Tensor& input);
    Tensor avg_pool_forward(const Tensor& input);
    Tensor max_pool_backward(const Tensor& output_gradient);
    Tensor avg_pool_backward(const Tensor& output_gradient);
};

class MaxPoolingLayer : public PoolingLayer {
public:
    explicit MaxPoolingLayer(int kernel_size, int stride = -1, int padding = 0)
        : PoolingLayer(PoolingType::MAX, kernel_size, stride, padding) {}
    
    MaxPoolingLayer(int kernel_h, int kernel_w, int stride_h, int stride_w,
                    int pad_h, int pad_w)
        : PoolingLayer(PoolingType::MAX, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) {}
};

class AvgPoolingLayer : public PoolingLayer {
public:
    explicit AvgPoolingLayer(int kernel_size, int stride = -1, int padding = 0)
        : PoolingLayer(PoolingType::AVERAGE, kernel_size, stride, padding) {}
    
    AvgPoolingLayer(int kernel_h, int kernel_w, int stride_h, int stride_w,
                    int pad_h, int pad_w)
        : PoolingLayer(PoolingType::AVERAGE, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) {}
};

} // namespace cnn
