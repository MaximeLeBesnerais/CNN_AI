#pragma once

#include "cnn/tensor.h"
#include <memory>
#include <string>

namespace cnn {

class Layer {
public:
    virtual ~Layer() = default;
    
    // Pure virtual functions that all layers must implement
    virtual Tensor forward(const Tensor& input, bool training = true) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
    
    // Optional methods
    virtual void update_weights(float learning_rate) {}
    virtual void set_training(bool training) { training_ = training; }
    virtual bool is_training() const { return training_; }
    virtual std::string name() const = 0;
    
    // Get parameters for optimizer
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_gradients() { return {}; }
    
protected:
    bool training_ = true;
};

using LayerPtr = std::unique_ptr<Layer>;

} // namespace cnn
