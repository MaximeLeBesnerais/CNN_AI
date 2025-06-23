#include <iostream>
#include <cassert>
#include <cmath>
#include "cnn/cnn.h"

// Import test functions from other test files
extern void run_tensor_tests();
extern void run_layer_tests();
extern void run_advanced_blocks_tests();  // New test suite

using namespace cnn;

void test_tensor_operations() {
    std::cout << "Testing tensor operations..." << std::endl;
    
    // Test basic tensor creation and operations
    Tensor a({2, 3});
    a.fill(2.0f);
    
    Tensor b({2, 3});
    b.fill(3.0f);
    
    auto c = a + b;
    assert(std::abs(c.at(0, 0) - 5.0f) < 1e-6f);
    
    // Test matrix multiplication
    Tensor x({2, 3});
    x.at(0, 0) = 1; x.at(0, 1) = 2; x.at(0, 2) = 3;
    x.at(1, 0) = 4; x.at(1, 1) = 5; x.at(1, 2) = 6;
    
    Tensor y({3, 2});
    y.at(0, 0) = 1; y.at(0, 1) = 2;
    y.at(1, 0) = 3; y.at(1, 1) = 4;
    y.at(2, 0) = 5; y.at(2, 1) = 6;
    
    auto z = x.matmul(y);
    assert(z.shape()[0] == 2 && z.shape()[1] == 2);
    assert(std::abs(z.at(0, 0) - 22.0f) < 1e-6f);  // 1*1 + 2*3 + 3*5 = 22
    
    std::cout << "Tensor operations test passed!" << std::endl;
}

void test_linear_layer() {
    std::cout << "Testing linear layer..." << std::endl;
    
    LinearLayer layer(3, 2);
    layer.zero_init();
    
    // Set simple weights for testing
    layer.weights().at(0, 0) = 1.0f; layer.weights().at(0, 1) = 0.0f;
    layer.weights().at(1, 0) = 0.0f; layer.weights().at(1, 1) = 1.0f;
    layer.weights().at(2, 0) = 0.0f; layer.weights().at(2, 1) = 0.0f;
    
    Tensor input({1, 3});
    input.at(0, 0) = 2.0f;
    input.at(0, 1) = 3.0f;
    input.at(0, 2) = 4.0f;
    
    auto output = layer.forward(input);
    assert(std::abs(output.at(0, 0) - 2.0f) < 1e-6f);
    assert(std::abs(output.at(0, 1) - 3.0f) < 1e-6f);
    
    std::cout << "Linear layer test passed!" << std::endl;
}

void test_activation_layers() {
    std::cout << "Testing activation layers..." << std::endl;
    
    ReLULayer relu;
    
    Tensor input({2, 2});
    input.at(0, 0) = -1.0f; input.at(0, 1) = 2.0f;
    input.at(1, 0) = -3.0f; input.at(1, 1) = 4.0f;
    
    auto output = relu.forward(input);
    assert(std::abs(output.at(0, 0) - 0.0f) < 1e-6f);
    assert(std::abs(output.at(0, 1) - 2.0f) < 1e-6f);
    assert(std::abs(output.at(1, 0) - 0.0f) < 1e-6f);
    assert(std::abs(output.at(1, 1) - 4.0f) < 1e-6f);
    
    std::cout << "Activation layers test passed!" << std::endl;
}

void test_simple_model() {
    std::cout << "Testing simple model..." << std::endl;
    
    Model model;
    model.add_layer<LinearLayer>(2, 3);
    model.add_layer<ReLULayer>();
    model.add_layer<LinearLayer>(3, 1);
    
    Tensor input({1, 2});
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    
    auto output = model.forward(input);
    assert(output.shape()[0] == 1 && output.shape()[1] == 1);
    
    std::cout << "Simple model test passed!" << std::endl;
}

void test_loss_functions() {
    std::cout << "Testing loss functions..." << std::endl;
    
    MeanSquaredError mse;
    
    Tensor predictions({2, 1});
    predictions.at(0, 0) = 1.0f;
    predictions.at(1, 0) = 2.0f;
    
    Tensor targets({2, 1});
    targets.at(0, 0) = 1.5f;
    targets.at(1, 0) = 1.5f;
    
    float loss = mse.compute(predictions, targets);
    std::cout << "Computed loss: " << loss << ", Expected: 0.125" << std::endl;
    // ((1-1.5)^2 + (2-1.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    assert(std::abs(loss - 0.25f) < 1e-6f);
    
    std::cout << "Loss functions test passed!" << std::endl;
}

int main() {
    try {
        test_tensor_operations();
        test_linear_layer();
        test_activation_layers();
        test_simple_model();
        test_loss_functions();
        
        std::cout << "\nAll tests passed! 🎉" << std::endl;
        
        // Print a simple model summary
        std::cout << "\nExample model summary:" << std::endl;
        Model example_model;
        example_model.add_layer<Conv2DLayer>(3, 32, 3, 3, 1, 1, 1, 1);  // explicit params
        example_model.add_layer<ReLULayer>();
        example_model.add_layer<MaxPoolingLayer>(2);
        example_model.add_layer<FlattenLayer>();
        example_model.add_layer<LinearLayer>(32 * 16 * 16, 10);  // Assuming 32x32 input
        example_model.summary();
        
        // Run all test suites
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Running comprehensive test suite..." << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        run_tensor_tests();
        run_layer_tests();
        run_advanced_blocks_tests();  // Test new advanced blocks
        
        std::cout << "\n🎉 All test suites completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
