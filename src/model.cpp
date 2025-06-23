#include "cnn/model.h"
#include <stdexcept>
#include <iostream>
#include <fstream>

namespace cnn {

void Model::add(LayerPtr layer) {
    layers_.push_back(std::move(layer));
}

Tensor Model::forward(const Tensor& input, bool training) {
    if (layers_.empty()) {
        throw std::runtime_error("Model has no layers");
    }
    
    Tensor current_input = input;
    for (auto& layer : layers_) {
        current_input = layer->forward(current_input, training);
    }
    
    return current_input;
}

Tensor Model::backward(const Tensor& output_gradient) {
    if (layers_.empty()) {
        throw std::runtime_error("Model has no layers");
    }
    
    Tensor current_gradient = output_gradient;
    
    // Backward pass in reverse order
    for (int i = layers_.size() - 1; i >= 0; --i) {
        current_gradient = layers_[i]->backward(current_gradient);
    }
    
    return current_gradient;
}

void Model::update_weights(float learning_rate) {
    for (auto& layer : layers_) {
        layer->update_weights(learning_rate);
    }
}

void Model::set_training(bool training) {
    for (auto& layer : layers_) {
        layer->set_training(training);
    }
}

const Layer& Model::get_layer(size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return *layers_[index];
}

std::vector<Tensor*> Model::get_parameters() {
    std::vector<Tensor*> all_params;
    for (auto& layer : layers_) {
        auto layer_params = layer->get_parameters();
        all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    return all_params;
}

std::vector<Tensor*> Model::get_gradients() {
    std::vector<Tensor*> all_grads;
    for (auto& layer : layers_) {
        auto layer_grads = layer->get_gradients();
        all_grads.insert(all_grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return all_grads;
}

void Model::summary() const {
    std::cout << "Model Summary:\n";
    std::cout << "=============\n";
    
    size_t total_params = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        auto params = layer->get_parameters();
        size_t layer_params = 0;
        
        for (const auto& param : params) {
            layer_params += param->size();
        }
        
        std::cout << "Layer " << i << " (" << layer->name() << "): " 
                  << layer_params << " parameters\n";
        total_params += layer_params;
    }
    
    std::cout << "=============\n";
    std::cout << "Total parameters: " << total_params << "\n";
}

void Model::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Save number of layers
    size_t num_layers = layers_.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Save each layer's parameters
    for (const auto& layer : layers_) {
        auto params = layer->get_parameters();
        size_t num_params = params.size();
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        
        for (const auto& param : params) {
            // Save shape
            size_t ndim = param->shape().size();
            file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            file.write(reinterpret_cast<const char*>(param->shape().data()), 
                      ndim * sizeof(int));
            
            // Save data
            size_t data_size = param->size();
            file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
            file.write(reinterpret_cast<const char*>(param->data()), 
                      data_size * sizeof(float));
        }
    }
    
    std::cout << "Model saved to " << filename << std::endl;
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read number of layers
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    if (num_layers != layers_.size()) {
        throw std::runtime_error("Model structure mismatch");
    }
    
    // Load each layer's parameters
    for (auto& layer : layers_) {
        auto params = layer->get_parameters();
        size_t num_params;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        
        if (num_params != params.size()) {
            throw std::runtime_error("Layer parameter count mismatch");
        }
        
        for (auto& param : params) {
            // Read shape
            size_t ndim;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            
            std::vector<int> shape(ndim);
            file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int));
            
            // Read data
            size_t data_size;
            file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            
            if (data_size != param->size()) {
                throw std::runtime_error("Parameter size mismatch");
            }
            
            file.read(reinterpret_cast<char*>(param->data()), data_size * sizeof(float));
        }
    }
    
    std::cout << "Model loaded from " << filename << std::endl;
}

} // namespace cnn
