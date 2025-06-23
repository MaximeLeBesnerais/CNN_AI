#include "cnn/tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace cnn {

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    
    strides_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

int Tensor::compute_flat_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    int flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_index += indices[i] * strides_[i];
    }
    return flat_index;
}

// Constructors
Tensor::Tensor() {}

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    compute_strides();
    int total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<int>& shape, float value) : shape_(shape) {
    compute_strides();
    int total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_.resize(total_size, value);
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) 
    : shape_(shape), data_(data) {
    compute_strides();
    int expected_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (data_.size() != expected_size) {
        throw std::invalid_argument("Data size doesn't match tensor shape");
    }
}

Tensor::Tensor(std::initializer_list<std::initializer_list<float>> data_2d) {
    if (data_2d.size() == 0) return;
    
    int rows = data_2d.size();
    int cols = data_2d.begin()->size();
    
    // Check all rows have same size
    for (const auto& row : data_2d) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same size");
        }
    }
    
    shape_ = {rows, cols};
    compute_strides();
    
    data_.reserve(rows * cols);
    for (const auto& row : data_2d) {
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

// Copy constructor and assignment
Tensor::Tensor(const Tensor& other) 
    : data_(other.data_), shape_(other.shape_), strides_(other.strides_) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
    }
    return *this;
}

// Move constructor and assignment
Tensor::Tensor(Tensor&& other) noexcept 
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}

// Element access
float& Tensor::operator()(const std::vector<int>& indices) {
    return data_[compute_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<int>& indices) const {
    return data_[compute_flat_index(indices)];
}

float& Tensor::at(int i) {
    if (shape_.size() != 1) throw std::invalid_argument("Tensor must be 1D");
    return data_[i];
}

const float& Tensor::at(int i) const {
    if (shape_.size() != 1) throw std::invalid_argument("Tensor must be 1D");
    return data_[i];
}

float& Tensor::at(int i, int j) {
    if (shape_.size() != 2) throw std::invalid_argument("Tensor must be 2D");
    return data_[i * strides_[0] + j * strides_[1]];
}

const float& Tensor::at(int i, int j) const {
    if (shape_.size() != 2) throw std::invalid_argument("Tensor must be 2D");
    return data_[i * strides_[0] + j * strides_[1]];
}

float& Tensor::at(int n, int c, int h, int w) {
    if (shape_.size() != 4) throw std::invalid_argument("Tensor must be 4D");
    return data_[n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3]];
}

const float& Tensor::at(int n, int c, int h, int w) const {
    if (shape_.size() != 4) throw std::invalid_argument("Tensor must be 4D");
    return data_[n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3]];
}

// Shape operations
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != size()) {
        throw std::invalid_argument("New shape must have same total size");
    }
    
    Tensor result(new_shape, data_);
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("transpose() without arguments only works for 2D tensors");
    }
    return transpose(0, 1);
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    if (dim1 < 0) dim1 += shape_.size();
    if (dim2 < 0) dim2 += shape_.size();
    
    if (dim1 >= shape_.size() || dim2 >= shape_.size() || dim1 < 0 || dim2 < 0) {
        throw std::out_of_range("Dimension indices out of range");
    }
    
    std::vector<int> new_shape = shape_;
    std::swap(new_shape[dim1], new_shape[dim2]);
    
    Tensor result(new_shape);
    
    // Copy data with transposed indices
    std::vector<int> indices(shape_.size(), 0);
    std::function<void(int)> copy_recursive = [&](int dim) {
        if (dim == shape_.size()) {
            std::vector<int> new_indices = indices;
            std::swap(new_indices[dim1], new_indices[dim2]);
            result(new_indices) = (*this)(indices);
            return;
        }
        
        for (int i = 0; i < shape_[dim]; ++i) {
            indices[dim] = i;
            copy_recursive(dim + 1);
        }
    };
    
    copy_recursive(0);
    return result;
}

Tensor Tensor::flatten() const {
    return reshape({size()});
}

// Element-wise operations
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

// Scalar operations
Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

// In-place operations
Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    for (int i = 0; i < size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    for (int i = 0; i < size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(float scalar) {
    for (int i = 0; i < size(); ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    for (int i = 0; i < size(); ++i) {
        data_[i] /= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(float scalar) {
    for (int i = 0; i < size(); ++i) {
        data_[i] /= scalar;
    }
    return *this;
}

// Matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Both tensors must be 2D for matrix multiplication");
    }
    
    int m = shape_[0];    // rows of A
    int k = shape_[1];    // cols of A / rows of B
    int n = other.shape_[1];  // cols of B
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }
    
    Tensor result({m, n});
    
    // Simple triple nested loop - can be optimized later
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += at(i, l) * other.at(l, j);
            }
            result.at(i, j) = sum;
        }
    }
    
    return result;
}

// Reduction operations
Tensor Tensor::sum(int axis) const {
    if (axis == -1) {
        // Sum all elements
        float total = std::accumulate(data_.begin(), data_.end(), 0.0f);
        return Tensor({1}, total);
    }
    
    if (axis < 0) axis += shape_.size();
    if (axis >= shape_.size()) {
        throw std::out_of_range("Axis out of range");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    if (new_shape.empty()) new_shape.push_back(1);
    
    Tensor result(new_shape);
    result.zero();
    
    // Iterate through all elements and accumulate
    std::vector<int> indices(shape_.size(), 0);
    std::function<void(int)> sum_recursive = [&](int dim) {
        if (dim == shape_.size()) {
            std::vector<int> result_indices;
            for (int i = 0; i < indices.size(); ++i) {
                if (i != axis) result_indices.push_back(indices[i]);
            }
            if (result_indices.empty()) result_indices.push_back(0);
            result(result_indices) += (*this)(indices);
            return;
        }
        
        for (int i = 0; i < shape_[dim]; ++i) {
            indices[dim] = i;
            sum_recursive(dim + 1);
        }
    };
    
    sum_recursive(0);
    return result;
}

// Missing max method implementation
Tensor Tensor::max(int axis) const {
    if (axis == -1) {
        // Max of all elements
        float max_val = *std::max_element(data_.begin(), data_.end());
        return Tensor({1}, max_val);
    }
    
    if (axis < 0) axis += shape_.size();
    if (axis >= shape_.size()) {
        throw std::out_of_range("Axis out of range");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    if (new_shape.empty()) new_shape.push_back(1);
    
    Tensor result(new_shape);
    
    // Initialize with very small values
    for (int i = 0; i < result.size(); ++i) {
        result.data()[i] = -std::numeric_limits<float>::infinity();
    }
    
    // Find max along axis
    std::vector<int> indices(shape_.size(), 0);
    std::function<void(int)> max_recursive = [&](int dim) {
        if (dim == shape_.size()) {
            std::vector<int> result_indices;
            for (int i = 0; i < indices.size(); ++i) {
                if (i != axis) result_indices.push_back(indices[i]);
            }
            if (result_indices.empty()) result_indices.push_back(0);
            
            float current_val = (*this)(indices);
            if (current_val > result(result_indices)) {
                result(result_indices) = current_val;
            }
            return;
        }
        
        for (int i = 0; i < shape_[dim]; ++i) {
            indices[dim] = i;
            max_recursive(dim + 1);
        }
    };
    
    max_recursive(0);
    return result;
}

Tensor Tensor::min(int axis) const {
    if (axis == -1) {
        // Min of all elements
        float min_val = *std::min_element(data_.begin(), data_.end());
        return Tensor({1}, min_val);
    }
    
    if (axis < 0) axis += shape_.size();
    if (axis >= shape_.size()) {
        throw std::out_of_range("Axis out of range");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    if (new_shape.empty()) new_shape.push_back(1);
    
    Tensor result(new_shape);
    
    // Initialize with very large values
    for (int i = 0; i < result.size(); ++i) {
        result.data()[i] = std::numeric_limits<float>::infinity();
    }
    
    // Find min along axis
    std::vector<int> indices(shape_.size(), 0);
    std::function<void(int)> min_recursive = [&](int dim) {
        if (dim == shape_.size()) {
            std::vector<int> result_indices;
            for (int i = 0; i < indices.size(); ++i) {
                if (i != axis) result_indices.push_back(indices[i]);
            }
            if (result_indices.empty()) result_indices.push_back(0);
            
            float current_val = (*this)(indices);
            if (current_val < result(result_indices)) {
                result(result_indices) = current_val;
            }
            return;
        }
        
        for (int i = 0; i < shape_[dim]; ++i) {
            indices[dim] = i;
            min_recursive(dim + 1);
        }
    };
    
    min_recursive(0);
    return result;
}

// Utility functions
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero() {
    fill(0.0f);
}

void Tensor::random_normal(float mean, float std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, std);
    
    for (auto& val : data_) {
        val = dis(gen);
    }
}

void Tensor::xavier_init(int fan_in, int fan_out) {
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    random_normal(0.0f, std);
}

void Tensor::he_init(int fan_in) {
    float std = std::sqrt(2.0f / fan_in);
    random_normal(0.0f, std);
}

// Activation functions
Tensor Tensor::relu() const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}

Tensor Tensor::leaky_relu(float alpha) const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] > 0 ? data_[i] : alpha * data_[i];
    }
    return result;
}

Tensor Tensor::relu_derivative() const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] > 0 ? 1.0f : 0.0f;
    }
    return result;
}

Tensor Tensor::leaky_relu_derivative(float alpha) const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] > 0 ? 1.0f : alpha;
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    return result;
}

Tensor Tensor::sigmoid_derivative() const {
    auto sig = sigmoid();
    return sig * (Tensor::ones(shape_) - sig);
}

Tensor Tensor::tanh_derivative() const {
    auto tanh_vals = tanh();
    return Tensor::ones(shape_) - tanh_vals * tanh_vals;
}

Tensor Tensor::softmax(int axis) const {
    // Simplified softmax implementation for 2D tensors (batch processing)
    if (shape_.size() == 2) {
        Tensor result(shape_);
        int batch_size = shape_[0];
        int num_classes = shape_[1];
        
        for (int b = 0; b < batch_size; ++b) {
            // Find max for numerical stability
            float max_val = at(b, 0);
            for (int c = 1; c < num_classes; ++c) {
                max_val = std::max(max_val, at(b, c));
            }
            
            // Compute exponentials and sum
            float sum_exp = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float exp_val = std::exp(at(b, c) - max_val);
                result.at(b, c) = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize
            for (int c = 0; c < num_classes; ++c) {
                result.at(b, c) /= sum_exp;
            }
        }
        
        return result;
    }
    
    // Fallback for other dimensions
    Tensor result(shape_);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = std::exp(data_[i]);
    }
    
    float sum_exp = std::accumulate(result.data_.begin(), result.data_.end(), 0.0f);
    for (int i = 0; i < size(); ++i) {
        result.data_[i] /= sum_exp;
    }
    
    return result;
}

// im2col implementation for convolution optimization
Tensor Tensor::im2col(int kernel_h, int kernel_w, int stride_h, int stride_w, 
                      int pad_h, int pad_w) const {
    if (shape_.size() != 4) {
        throw std::invalid_argument("im2col requires 4D tensor (N, C, H, W)");
    }
    
    int N = shape_[0];  // batch size
    int C = shape_[1];  // channels
    int H = shape_[2];  // height
    int W = shape_[3];  // width
    
    int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Output tensor: (N * out_h * out_w, C * kernel_h * kernel_w)
    Tensor result({N * out_h * out_w, C * kernel_h * kernel_w});
    result.zero();
    
    for (int n = 0; n < N; ++n) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                int col_idx = n * out_h * out_w + y * out_w + x;
                
                for (int c = 0; c < C; ++c) {
                    for (int ky = 0; ky < kernel_h; ++ky) {
                        for (int kx = 0; kx < kernel_w; ++kx) {
                            int row_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                            
                            int h_idx = y * stride_h + ky - pad_h;
                            int w_idx = x * stride_w + kx - pad_w;
                            
                            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                                result.at(col_idx, row_idx) = at(n, c, h_idx, w_idx);
                            }
                            // else: padding value (already zero)
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// col2im implementation
Tensor Tensor::col2im(const Tensor& cols, const std::vector<int>& input_shape,
                     int kernel_h, int kernel_w, int stride_h, int stride_w,
                     int pad_h, int pad_w) {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("col2im requires 4D input shape (N, C, H, W)");
    }
    
    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];
    
    int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;
    
    Tensor result(input_shape);
    result.zero();
    
    for (int n = 0; n < N; ++n) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                int col_idx = n * out_h * out_w + y * out_w + x;
                
                for (int c = 0; c < C; ++c) {
                    for (int ky = 0; ky < kernel_h; ++ky) {
                        for (int kx = 0; kx < kernel_w; ++kx) {
                            int row_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                            
                            int h_idx = y * stride_h + ky - pad_h;
                            int w_idx = x * stride_w + kx - pad_w;
                            
                            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                                result.at(n, c, h_idx, w_idx) += cols.at(col_idx, row_idx);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// Static factory methods
Tensor Tensor::zeros(const std::vector<int>& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const std::vector<int>& shape) {
    return Tensor(shape, 1.0f);
}

// Debugging and printing
void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

std::string Tensor::to_string() const {
    std::stringstream ss;
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i];
    }
    ss << "], data=[\n";
    
    if (shape_.size() <= 2) {
        for (int i = 0; i < std::min(size(), 100); ++i) {
            if (i > 0) ss << ", ";
            ss << std::fixed << std::setprecision(4) << data_[i];
        }
        if (size() > 100) ss << "...";
    } else {
        ss << "..."; // For higher dimensional tensors, just show shape
    }
    
    ss << "])";
    return ss.str();
}

// Non-member operators
Tensor operator+(float scalar, const Tensor& tensor) {
    return tensor + scalar;
}

Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor result(tensor.shape());
    for (int i = 0; i < tensor.size(); ++i) {
        result.data()[i] = scalar - tensor.data()[i];
    }
    return result;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}

} // namespace cnn
