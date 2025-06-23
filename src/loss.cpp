#include "cnn/loss.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace cnn {

// Mean Squared Error Implementation
float MeanSquaredError::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    float sum_squared_error = 0.0f;
    for (int i = 0; i < predictions.size(); ++i) {
        float diff = predictions.data()[i] - targets.data()[i];
        sum_squared_error += diff * diff;
    }
    
    return sum_squared_error / predictions.size();
}

Tensor MeanSquaredError::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    // Gradient of MSE: 2 * (predictions - targets) / batch_size
    Tensor grad = (predictions - targets) * (2.0f / predictions.size());
    return grad;
}

// Cross Entropy Loss Implementation
float CrossEntropyLoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    // Assuming predictions are already softmax probabilities
    // and targets are one-hot encoded
    float loss = 0.0f;
    int batch_size = predictions.shape()[0];
    
    for (int i = 0; i < predictions.size(); ++i) {
        // Clamp predictions to prevent log(0)
        float pred = std::max(epsilon_, std::min(1.0f - epsilon_, predictions.data()[i]));
        loss -= targets.data()[i] * std::log(pred);
    }
    
    return loss / batch_size;
}

Tensor CrossEntropyLoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    // For softmax + cross-entropy, the gradient simplifies to: predictions - targets
    int batch_size = predictions.shape()[0];
    return (predictions - targets) * (1.0f / batch_size);
}

// Binary Cross Entropy Loss Implementation
float BinaryCrossEntropyLoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    float loss = 0.0f;
    
    for (int i = 0; i < predictions.size(); ++i) {
        float pred = std::max(epsilon_, std::min(1.0f - epsilon_, predictions.data()[i]));
        float target = targets.data()[i];
        
        loss -= target * std::log(pred) + (1.0f - target) * std::log(1.0f - pred);
    }
    
    return loss / predictions.size();
}

Tensor BinaryCrossEntropyLoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    Tensor grad(predictions.shape());
    
    for (int i = 0; i < predictions.size(); ++i) {
        float pred = std::max(epsilon_, std::min(1.0f - epsilon_, predictions.data()[i]));
        float target = targets.data()[i];
        
        grad.data()[i] = (pred - target) / (pred * (1.0f - pred)) / predictions.size();
    }
    
    return grad;
}

} // namespace cnn
