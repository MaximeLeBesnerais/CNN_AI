#pragma once

#include "cnn/tensor.h"
#include <vector>
#include <memory>

namespace cnn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual std::string name() const = 0;
    
protected:
    std::vector<Tensor*> parameters_;
    std::vector<Tensor*> gradients_;
    float learning_rate_;
    
public:
    Optimizer(const std::vector<Tensor*>& parameters, 
              const std::vector<Tensor*>& gradients, 
              float learning_rate)
        : parameters_(parameters), gradients_(gradients), learning_rate_(learning_rate) {
        
        if (parameters_.size() != gradients_.size()) {
            throw std::invalid_argument("Parameters and gradients must have same size");
        }
    }
    
    float learning_rate() const { return learning_rate_; }
    void set_learning_rate(float lr) { learning_rate_ = lr; }
};

class SGDOptimizer : public Optimizer {
private:
    float momentum_;
    std::vector<Tensor> velocity_;
    
public:
    SGDOptimizer(const std::vector<Tensor*>& parameters,
                 const std::vector<Tensor*>& gradients,
                 float learning_rate,
                 float momentum = 0.0f);
    
    void step() override;
    void zero_grad() override;
    std::string name() const override { return "SGD"; }
    
    float momentum() const { return momentum_; }
};

class AdamOptimizer : public Optimizer {
private:
    float beta1_;
    float beta2_;
    float epsilon_;
    std::vector<Tensor> m_;  // First moment
    std::vector<Tensor> v_;  // Second moment
    int t_;  // Time step
    
public:
    AdamOptimizer(const std::vector<Tensor*>& parameters,
                  const std::vector<Tensor*>& gradients,
                  float learning_rate,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f);
    
    void step() override;
    void zero_grad() override;
    std::string name() const override { return "Adam"; }
    
    float beta1() const { return beta1_; }
    float beta2() const { return beta2_; }
};

class RMSPropOptimizer : public Optimizer {
private:
    float alpha_;
    float epsilon_;
    std::vector<Tensor> squared_avg_;
    
public:
    RMSPropOptimizer(const std::vector<Tensor*>& parameters,
                     const std::vector<Tensor*>& gradients,
                     float learning_rate,
                     float alpha = 0.99f,
                     float epsilon = 1e-8f);
    
    void step() override;
    void zero_grad() override;
    std::string name() const override { return "RMSProp"; }
    
    float alpha() const { return alpha_; }
};

using OptimizerPtr = std::unique_ptr<Optimizer>;

} // namespace cnn
