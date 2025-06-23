#pragma once

#include "cnn/layer.h"
#include <vector>
#include <memory>

namespace cnn {

class Model {
private:
    std::vector<LayerPtr> layers_;
    
public:
    Model() = default;
    ~Model() = default;
    
    // Non-copyable but movable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;
    
    // Add layers to the model
    void add(LayerPtr layer);
    
    template<typename LayerType, typename... Args>
    void add_layer(Args&&... args) {
        add(std::make_unique<LayerType>(std::forward<Args>(args)...));
    }
    
    // Forward pass through all layers
    Tensor forward(const Tensor& input, bool training = true);
    
    // Backward pass through all layers
    Tensor backward(const Tensor& output_gradient);
    
    // Update all layer weights
    void update_weights(float learning_rate);
    
    // Set training mode for all layers
    void set_training(bool training);
    
    // Get model information
    size_t num_layers() const { return layers_.size(); }
    const Layer& get_layer(size_t index) const;
    
    // Get all parameters and gradients
    std::vector<Tensor*> get_parameters();
    std::vector<Tensor*> get_gradients();
    
    // Model summary
    void summary() const;
    
    // Save/load model (simplified)
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};

} // namespace cnn
