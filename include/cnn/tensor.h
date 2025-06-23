#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>

namespace cnn {

class Tensor {
private:
    std::vector<float> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    
    void compute_strides();
    int compute_flat_index(const std::vector<int>& indices) const;

public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, float value);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);
    Tensor(std::initializer_list<std::initializer_list<float>> data_2d);
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor() = default;
    
    // Accessors
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int size() const { return data_.size(); }
    int ndim() const { return shape_.size(); }
    
    // Element access
    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;
    float& at(int i);
    const float& at(int i) const;
    float& at(int i, int j);
    const float& at(int i, int j) const;
    float& at(int n, int c, int h, int w);
    const float& at(int n, int c, int h, int w) const;
    
    // Raw data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    // Shape operations
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor transpose() const;
    Tensor transpose(int dim1, int dim2) const;
    Tensor flatten() const;
    Tensor squeeze() const;
    Tensor unsqueeze(int dim) const;
    
    // Element-wise operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    Tensor& operator+=(float scalar);
    Tensor& operator-=(float scalar);
    Tensor& operator*=(float scalar);
    Tensor& operator/=(float scalar);
    
    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor sum(int axis = -1) const;
    Tensor mean(int axis = -1) const;
    Tensor max(int axis = -1) const;
    Tensor min(int axis = -1) const;
    
    // Broadcasting
    bool is_broadcastable(const Tensor& other) const;
    std::pair<Tensor, Tensor> broadcast(const Tensor& other) const;
    
    // Utility functions
    void fill(float value);
    void zero();
    void random_normal(float mean = 0.0f, float std = 1.0f);
    void xavier_init(int fan_in, int fan_out);
    void he_init(int fan_in);
    
    // Activation functions (applied element-wise)
    Tensor relu() const;
    Tensor leaky_relu(float alpha = 0.01f) const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int axis = -1) const;
    
    // Derivatives of activation functions
    Tensor relu_derivative() const;
    Tensor leaky_relu_derivative(float alpha = 0.01f) const;
    Tensor sigmoid_derivative() const;
    Tensor tanh_derivative() const;
    
    // Padding operations
    Tensor pad(const std::vector<std::pair<int, int>>& padding, float value = 0.0f) const;
    
    // Convolution helper operations
    Tensor im2col(int kernel_h, int kernel_w, int stride_h = 1, int stride_w = 1, 
                  int pad_h = 0, int pad_w = 0) const;
    static Tensor col2im(const Tensor& cols, const std::vector<int>& input_shape,
                        int kernel_h, int kernel_w, int stride_h = 1, int stride_w = 1,
                        int pad_h = 0, int pad_w = 0);
    
    // Debugging and printing
    void print() const;
    std::string to_string() const;
    
    // Static factory methods
    static Tensor zeros(const std::vector<int>& shape);
    static Tensor ones(const std::vector<int>& shape);
    static Tensor random(const std::vector<int>& shape, float min = 0.0f, float max = 1.0f);
    static Tensor arange(float start, float stop, float step = 1.0f);
    static Tensor linspace(float start, float stop, int num);
};

// Non-member operators
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator/(float scalar, const Tensor& tensor);

// Stream output
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace cnn
