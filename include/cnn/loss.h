#pragma once

#include "cnn/tensor.h"

namespace cnn {

class LossFunction {
public:
    virtual ~LossFunction() = default;
    
    virtual float compute(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) = 0;
    virtual std::string name() const = 0;
};

class MeanSquaredError : public LossFunction {
public:
    float compute(const Tensor& predictions, const Tensor& targets) override;
    Tensor gradient(const Tensor& predictions, const Tensor& targets) override;
    std::string name() const override { return "MSE"; }
};

class CrossEntropyLoss : public LossFunction {
private:
    float epsilon_;  // Small value to prevent log(0)
    
public:
    CrossEntropyLoss(float epsilon = 1e-8f) : epsilon_(epsilon) {}
    
    float compute(const Tensor& predictions, const Tensor& targets) override;
    Tensor gradient(const Tensor& predictions, const Tensor& targets) override;
    std::string name() const override { return "CrossEntropy"; }
};

class BinaryCrossEntropyLoss : public LossFunction {
private:
    float epsilon_;
    
public:
    BinaryCrossEntropyLoss(float epsilon = 1e-8f) : epsilon_(epsilon) {}
    
    float compute(const Tensor& predictions, const Tensor& targets) override;
    Tensor gradient(const Tensor& predictions, const Tensor& targets) override;
    std::string name() const override { return "BinaryCrossEntropy"; }
};

using LossFunctionPtr = std::unique_ptr<LossFunction>;

} // namespace cnn
