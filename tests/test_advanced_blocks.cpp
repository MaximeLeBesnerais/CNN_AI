#include "cnn/cnn.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace cnn;

bool test_depthwise_conv_basic() {
    std::cout << "Testing DepthwiseConv2D basic functionality..." << std::endl;
    
    try {
        DepthwiseConv2DLayer layer(3, 3, 3, 1, 1, 1, 1);
        
        // Test forward pass
        Tensor input({1, 3, 8, 8});
        float* input_data = input.data();
        for (size_t i = 0; i < input.size(); ++i) {
            input_data[i] = (float)(i % 10) / 10.0f;
        }
        
        Tensor output = layer.forward(input, true);
        auto output_shape = output.shape();
        
        // Check output shape
        assert(output_shape.size() == 4);
        assert(output_shape[0] == 1);  // batch
        assert(output_shape[1] == 3);  // channels (unchanged)
        assert(output_shape[2] == 8);  // height (with padding)
        assert(output_shape[3] == 8);  // width (with padding)
        
        // Test backward pass
        Tensor grad_output({1, 3, 8, 8});
        float* grad_data = grad_output.data();
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_data[i] = 1.0f;
        }
        
        Tensor grad_input = layer.backward(grad_output);
        auto grad_shape = grad_input.shape();
        
        // Check gradient shape matches input
        assert(grad_shape == input.shape());
        
        std::cout << "  ✓ Basic functionality test passed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_residual_block_basic() {
    std::cout << "Testing ResidualBlock basic functionality..." << std::endl;
    
    try {
        // Test same dimensions (no shortcut conv needed)
        ResidualBlock block1(32, 32, 1);
        
        Tensor input1({2, 32, 16, 16});
        float* input1_data = input1.data();
        for (size_t i = 0; i < input1.size(); ++i) {
            input1_data[i] = (float)(i % 100) / 100.0f;
        }
        
        Tensor output1 = block1.forward(input1, true);
        assert(output1.shape() == input1.shape());
        
        // Test different dimensions (shortcut conv needed)
        ResidualBlock block2(32, 64, 2);
        
        Tensor input2({2, 32, 16, 16});
        float* input2_data = input2.data();
        for (size_t i = 0; i < input2.size(); ++i) {
            input2_data[i] = (float)(i % 100) / 100.0f;
        }
        
        Tensor output2 = block2.forward(input2, true);
        auto output2_shape = output2.shape();
        
        assert(output2_shape[0] == 2);   // batch unchanged
        assert(output2_shape[1] == 64);  // channels changed
        assert(output2_shape[2] == 8);   // height halved (stride=2)
        assert(output2_shape[3] == 8);   // width halved (stride=2)
        
        // Test backward pass
        Tensor grad_output({2, 64, 8, 8});
        float* grad_data = grad_output.data();
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_data[i] = 1.0f;
        }
        
        Tensor grad_input = block2.backward(grad_output);
        assert(grad_input.shape() == input2.shape());
        
        std::cout << "  ✓ Basic functionality test passed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_inception_module_basic() {
    std::cout << "Testing InceptionModule basic functionality..." << std::endl;
    
    try {
        InceptionModule module(128, 64, 96, 16, 32);
        
        Tensor input({1, 128, 32, 32});
        float* input_data = input.data();
        for (size_t i = 0; i < input.size(); ++i) {
            input_data[i] = (float)(i % 1000) / 1000.0f;
        }
        
        Tensor output = module.forward(input, true);
        auto output_shape = output.shape();
        
        // Check output shape
        int expected_channels = 64 + 96 + 16 + 32; // Sum of all branch channels
        assert(output_shape[0] == 1);                // batch unchanged
        assert(output_shape[1] == expected_channels); // concatenated channels
        assert(output_shape[2] == 32);               // height unchanged
        assert(output_shape[3] == 32);               // width unchanged
        
        // Test backward pass
        Tensor grad_output(output_shape);
        float* grad_data = grad_output.data();
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_data[i] = 1.0f;
        }
        
        Tensor grad_input = module.backward(grad_output);
        assert(grad_input.shape() == input.shape());
        
        std::cout << "  ✓ Basic functionality test passed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_bottleneck_block_basic() {
    std::cout << "Testing BottleneckBlock basic functionality..." << std::endl;
    
    try {
        // Test same input/output dimensions
        BottleneckBlock block1(256, 64, 256, 1);
        
        Tensor input1({1, 256, 16, 16});
        float* input1_data = input1.data();
        for (size_t i = 0; i < input1.size(); ++i) {
            input1_data[i] = (float)(i % 500) / 500.0f;
        }
        
        Tensor output1 = block1.forward(input1, true);
        assert(output1.shape() == input1.shape());
        
        // Test different dimensions
        BottleneckBlock block2(128, 128, 512, 2);
        
        Tensor input2({1, 128, 32, 32});
        float* input2_data = input2.data();
        for (size_t i = 0; i < input2.size(); ++i) {
            input2_data[i] = (float)(i % 500) / 500.0f;
        }
        
        Tensor output2 = block2.forward(input2, true);
        auto output2_shape = output2.shape();
        
        assert(output2_shape[0] == 1);   // batch unchanged
        assert(output2_shape[1] == 512); // channels expanded
        assert(output2_shape[2] == 16);  // height halved
        assert(output2_shape[3] == 16);  // width halved
        
        // Test backward pass
        Tensor grad_output(output2_shape);
        float* grad_data = grad_output.data();
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_data[i] = 1.0f;
        }
        
        Tensor grad_input = block2.backward(grad_output);
        assert(grad_input.shape() == input2.shape());
        
        std::cout << "  ✓ Basic functionality test passed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_gradient_flow() {
    std::cout << "Testing gradient flow through advanced blocks..." << std::endl;
    
    try {
        // Create a simple model with all advanced blocks
        Model model;
        
        model.add_layer<Conv2DLayer>(3, 32, 3, 3, 1, 1, 1, 1);
        model.add_layer<ReLULayer>();
        model.add_layer<DepthwiseConv2DLayer>(32, 3, 3, 1, 1, 1, 1);
        model.add_layer<ReLULayer>();
        model.add_layer<ResidualBlock>(32, 32, 1);
        model.add_layer<AvgPoolingLayer>(4);
        model.add_layer<FlattenLayer>();
        model.add_layer<LinearLayer>(32 * 2 * 2, 10);
        
        // Forward pass
        Tensor input({2, 3, 8, 8});
        float* input_data = input.data();
        for (size_t i = 0; i < input.size(); ++i) {
            input_data[i] = (float)rand() / RAND_MAX;
        }
        
        Tensor output = model.forward(input, true);
        
        // Backward pass
        Tensor grad_output({2, 10});
        float* grad_data = grad_output.data();
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_data[i] = 1.0f;
        }
        
        Tensor grad_input = model.backward(grad_output);
        
        // Check that gradients are computed (non-zero)
        float* grad_input_data = grad_input.data();
        bool has_nonzero_grad = false;
        for (size_t i = 0; i < grad_input.size(); ++i) {
            if (std::abs(grad_input_data[i]) > 1e-10) {
                has_nonzero_grad = true;
                break;
            }
        }
        
        assert(has_nonzero_grad);
        assert(grad_input.shape() == input.shape());
        
        std::cout << "  ✓ Gradient flow test passed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_performance_comparison() {
    std::cout << "Testing performance comparison: Regular vs Depthwise Conv..." << std::endl;
    
    try {
        const int channels = 64;
        const int input_size = 32;
        const int kernel_size = 3;
        
        // Regular convolution
        Conv2DLayer regular_conv(channels, channels, kernel_size, kernel_size, 1, 1, 1, 1);
        
        // Depthwise + Pointwise (separable convolution)
        DepthwiseConv2DLayer depthwise_conv(channels, kernel_size, kernel_size, 1, 1, 1, 1);
        Conv2DLayer pointwise_conv(channels, channels, 1, 1, 1, 1, 0, 0);
        
        Tensor input({1, channels, input_size, input_size});
        float* input_data = input.data();
        for (size_t i = 0; i < input.size(); ++i) {
            input_data[i] = (float)rand() / RAND_MAX;
        }
        
        // Time regular convolution
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            Tensor output = regular_conv.forward(input, true);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10;
        
        // Time depthwise separable convolution
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            Tensor temp = depthwise_conv.forward(input, true);
            Tensor output = pointwise_conv.forward(temp, true);
        }
        end = std::chrono::high_resolution_clock::now();
        auto separable_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10;
        
        std::cout << "  Regular conv time: " << regular_time << " μs" << std::endl;
        std::cout << "  Separable conv time: " << separable_time << " μs" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (float)regular_time / separable_time << "x" << std::endl;
        
        std::cout << "  ✓ Performance comparison completed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_architectural_patterns() {
    std::cout << "Testing real architectural patterns..." << std::endl;
    
    try {
        // Test MobileNet-like architecture
        Model mobilenet_style;
        mobilenet_style.add_layer<Conv2DLayer>(3, 32, 3, 3, 2, 2, 1, 1);
        mobilenet_style.add_layer<ReLULayer>();
        
        // Depthwise separable blocks
        for (int i = 0; i < 3; ++i) {
            int in_ch = (i == 0) ? 32 : 64;
            mobilenet_style.add_layer<DepthwiseConv2DLayer>(in_ch, 3, 3, 1, 1, 1, 1);
            mobilenet_style.add_layer<ReLULayer>();
            mobilenet_style.add_layer<Conv2DLayer>(in_ch, 64, 1, 1, 1, 1, 0, 0);
            mobilenet_style.add_layer<ReLULayer>();
        }
        
        mobilenet_style.add_layer<AvgPoolingLayer>(8);
        mobilenet_style.add_layer<FlattenLayer>();
        mobilenet_style.add_layer<LinearLayer>(64, 10);
        
        // Test ResNet-like architecture
        Model resnet_style;
        resnet_style.add_layer<Conv2DLayer>(3, 64, 7, 7, 2, 2, 3, 3);
        resnet_style.add_layer<ReLULayer>();
        resnet_style.add_layer<MaxPoolingLayer>(3, 3, 2, 2, 1, 1);
        
        // Residual blocks
        resnet_style.add_layer<ResidualBlock>(64, 64, 1);
        resnet_style.add_layer<ResidualBlock>(64, 128, 2);
        resnet_style.add_layer<ResidualBlock>(128, 128, 1);
        
        resnet_style.add_layer<AvgPoolingLayer>(7);
        resnet_style.add_layer<FlattenLayer>();
        resnet_style.add_layer<LinearLayer>(128, 10);
        
        // Test both architectures
        Tensor input({1, 3, 32, 32});
        float* input_data = input.data();
        for (size_t i = 0; i < input.size(); ++i) {
            input_data[i] = (float)rand() / RAND_MAX;
        }
        
        Tensor mobilenet_output = mobilenet_style.forward(input, false);
        Tensor resnet_output = resnet_style.forward(input, false);
        
        assert(mobilenet_output.shape()[1] == 10);
        assert(resnet_output.shape()[1] == 10);
        
        std::cout << "  ✓ MobileNet-style architecture works" << std::endl;
        std::cout << "  ✓ ResNet-style architecture works" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "  ✗ Test failed: " << e.what() << std::endl;
        return false;
    }
}

void run_advanced_blocks_tests() {
    std::cout << "\n=== Advanced Blocks Test Suite ===" << std::endl;
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    std::vector<std::pair<std::string, std::function<bool()>>> tests = {
        {"DepthwiseConv2D Basic", test_depthwise_conv_basic},
        {"ResidualBlock Basic", test_residual_block_basic},
        {"InceptionModule Basic", test_inception_module_basic},
        {"BottleneckBlock Basic", test_bottleneck_block_basic},
        {"Gradient Flow", test_gradient_flow},
        {"Performance Comparison", test_performance_comparison},
        {"Architectural Patterns", test_architectural_patterns}
    };
    
    for (auto& test : tests) {
        total_tests++;
        std::cout << "\nRunning " << test.first << "..." << std::endl;
        if (test.second()) {
            passed_tests++;
        }
    }
    
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Passed: " << passed_tests << "/" << total_tests << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
              << (100.0f * passed_tests / total_tests) << "%" << std::endl;
    
    if (passed_tests == total_tests) {
        std::cout << "🎉 All advanced blocks tests passed!" << std::endl;
    } else {
        std::cout << "⚠️  Some tests failed. Check implementation." << std::endl;
    }
}
