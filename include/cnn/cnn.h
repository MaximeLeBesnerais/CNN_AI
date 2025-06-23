#pragma once

// Core components
#include "cnn/tensor.h"
#include "cnn/layer.h"
#include "cnn/model.h"

// Layers
#include "cnn/linear.h"
#include "cnn/conv2d.h"
#include "cnn/activation.h"
#include "cnn/pooling.h"
#include "cnn/dropout.h"
#include "cnn/layers.h"  // BatchNorm, Flatten

// Advanced architectural blocks
#include "cnn/blocks.h"  // ResidualBlock, InceptionModule, DepthwiseConv2D, BottleneckBlock

// Training components
#include "cnn/optimizer.h"
#include "cnn/loss.h"

namespace cnn {

// Convenience functions for creating optimizers
inline std::unique_ptr<SGDOptimizer> make_sgd_optimizer(
    const std::vector<Tensor*>& parameters,
    const std::vector<Tensor*>& gradients,
    float learning_rate,
    float momentum = 0.0f) {
    return std::make_unique<SGDOptimizer>(parameters, gradients, learning_rate, momentum);
}

inline std::unique_ptr<AdamOptimizer> make_adam_optimizer(
    const std::vector<Tensor*>& parameters,
    const std::vector<Tensor*>& gradients,
    float learning_rate,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float epsilon = 1e-8f) {
    return std::make_unique<AdamOptimizer>(parameters, gradients, learning_rate, beta1, beta2, epsilon);
}

inline std::unique_ptr<RMSPropOptimizer> make_rmsprop_optimizer(
    const std::vector<Tensor*>& parameters,
    const std::vector<Tensor*>& gradients,
    float learning_rate,
    float alpha = 0.99f,
    float epsilon = 1e-8f) {
    return std::make_unique<RMSPropOptimizer>(parameters, gradients, learning_rate, alpha, epsilon);
}

// Convenience functions for creating loss functions
inline std::unique_ptr<MeanSquaredError> make_mse_loss() {
    return std::make_unique<MeanSquaredError>();
}

inline std::unique_ptr<CrossEntropyLoss> make_crossentropy_loss(float epsilon = 1e-8f) {
    return std::make_unique<CrossEntropyLoss>(epsilon);
}

inline std::unique_ptr<BinaryCrossEntropyLoss> make_bce_loss(float epsilon = 1e-8f) {
    return std::make_unique<BinaryCrossEntropyLoss>(epsilon);
}

} // namespace cnn
