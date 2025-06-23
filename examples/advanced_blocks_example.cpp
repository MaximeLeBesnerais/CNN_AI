#include "cnn/cnn.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace cnn;

void test_depthwise_convolution() {
    std::cout << "\n=== Testing Depthwise Convolution ===" << std::endl;
    
    // Create a simple model with depthwise convolution
    Model model;
    
    // Standard conv -> Depthwise conv -> Pointwise conv (MobileNet-style)
    model.add_layer<Conv2DLayer>(3, 32, 3, 3, 2, 2, 1, 1);    // Standard conv
    model.add_layer<ReLULayer>();
    model.add_layer<DepthwiseConv2DLayer>(32, 3, 3, 1, 1, 1, 1); // Depthwise conv
    model.add_layer<ReLULayer>();
    model.add_layer<Conv2DLayer>(32, 64, 1, 1, 1, 1, 0, 0);   // Pointwise conv
    model.add_layer<ReLULayer>();
    
    std::cout << "MobileNet-style Depthwise Separable Convolution Model:" << std::endl;
    
    // Test forward pass
    Tensor input({1, 3, 64, 64});  // Single RGB image
    float* input_data = input.data();
    for (int i = 0; i < input.size(); ++i) {
        input_data[i] = (float)rand() / RAND_MAX;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] 
              << ", " << input.shape()[2] << ", " << input.shape()[3] << "]" << std::endl;
    std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] 
              << ", " << output.shape()[2] << ", " << output.shape()[3] << "]" << std::endl;
    std::cout << "Forward pass time: " << duration.count() << " microseconds" << std::endl;
}

void test_residual_block() {
    std::cout << "\n=== Testing Residual Block ===" << std::endl;
    
    // Create a model with residual blocks (ResNet-style)
    Model model;
    
    model.add_layer<Conv2DLayer>(3, 16, 3, 3, 1, 1, 1, 1);    // Initial conv
    model.add_layer<ReLULayer>();
    
    // Stack of residual blocks
    model.add_layer<ResidualBlock>(16, 16, 1);  // Same dimensions
    model.add_layer<ResidualBlock>(16, 32, 2);  // Downsample + increase channels
    model.add_layer<ResidualBlock>(32, 32, 1);  // Same dimensions
    model.add_layer<ResidualBlock>(32, 64, 2);  // Downsample + increase channels
    
    model.add_layer<AvgPoolingLayer>(8);        // Global average pooling
    model.add_layer<FlattenLayer>();
    model.add_layer<LinearLayer>(64, 10);       // Classification
    model.add_layer<SoftmaxLayer>();
    
    std::cout << "ResNet-style Model with Residual Blocks:" << std::endl;
    
    // Test forward pass
    Tensor input({2, 3, 32, 32});  // Batch of 2 RGB images
    float* input_data = input.data();
    for (int i = 0; i < input.size(); ++i) {
        input_data[i] = (float)rand() / RAND_MAX;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] 
              << ", " << input.shape()[2] << ", " << input.shape()[3] << "]" << std::endl;
    std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
    std::cout << "Forward pass time: " << duration.count() << " microseconds" << std::endl;
    
    // Test backward pass
    Tensor loss_grad({2, 10});
    float* loss_data = loss_grad.data();
    for (int i = 0; i < loss_grad.size(); ++i) {
        loss_data[i] = (float)rand() / RAND_MAX;
    }
    
    start = std::chrono::high_resolution_clock::now();
    Tensor input_grad = model.backward(loss_grad);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Backward pass time: " << duration.count() << " microseconds" << std::endl;
}

void test_inception_module() {
    std::cout << "\n=== Testing Inception Module ===" << std::endl;
    
    // Create a model with Inception modules
    Model model;
    
    model.add_layer<Conv2DLayer>(3, 64, 7, 7, 2, 2, 3, 3);    // Initial conv
    model.add_layer<ReLULayer>();
    model.add_layer<MaxPoolingLayer>(3, 3, 2, 2, 1, 1);       // Max pooling
    
    // Stack of Inception modules
    model.add_layer<InceptionModule>(64, 64, 128, 32, 32);     // Mixed layer
    model.add_layer<InceptionModule>(256, 128, 192, 96, 64);   // Another mixed layer
    
    model.add_layer<AvgPoolingLayer>(7);        // Global average pooling
    model.add_layer<FlattenLayer>();
    model.add_layer<LinearLayer>(512, 1000);    // ImageNet-style classification
    model.add_layer<SoftmaxLayer>();
    
    std::cout << "GoogLeNet/Inception-style Model:" << std::endl;
    
    // Test forward pass with larger input (like ImageNet)
    Tensor input({1, 3, 224, 224});  // Single ImageNet-sized image
    float* input_data = input.data();
    for (int i = 0; i < input.size(); ++i) {
        input_data[i] = (float)rand() / RAND_MAX;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] 
              << ", " << input.shape()[2] << ", " << input.shape()[3] << "]" << std::endl;
    std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
    std::cout << "Forward pass time: " << duration.count() << " milliseconds" << std::endl;
}

void test_bottleneck_block() {
    std::cout << "\n=== Testing Bottleneck Block ===" << std::endl;
    
    // Create a model with bottleneck blocks (ResNet-50 style)
    Model model;
    
    model.add_layer<Conv2DLayer>(3, 64, 7, 7, 2, 2, 3, 3);    // Initial conv
    model.add_layer<BatchNormLayer>(64);
    model.add_layer<ReLULayer>();
    model.add_layer<MaxPoolingLayer>(3, 3, 2, 2, 1, 1);       // Max pooling
    
    // Stack of bottleneck blocks
    model.add_layer<BottleneckBlock>(64, 64, 256, 1);   // First bottleneck
    model.add_layer<BottleneckBlock>(256, 64, 256, 1);  // Same dimensions
    model.add_layer<BottleneckBlock>(256, 128, 512, 2); // Downsample
    model.add_layer<BottleneckBlock>(512, 128, 512, 1); // Same dimensions
    
    model.add_layer<AvgPoolingLayer>(7);        // Global average pooling
    model.add_layer<FlattenLayer>();
    model.add_layer<LinearLayer>(512, 1000);    // Classification
    model.add_layer<SoftmaxLayer>();
    
    std::cout << "ResNet-50 style Model with Bottleneck Blocks:" << std::endl;
    
    // Test forward pass
    Tensor input({1, 3, 224, 224});  // ImageNet-sized input
    float* input_data = input.data();
    for (int i = 0; i < input.size(); ++i) {
        input_data[i] = (float)rand() / RAND_MAX;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] 
              << ", " << input.shape()[2] << ", " << input.shape()[3] << "]" << std::endl;
    std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
    std::cout << "Forward pass time: " << duration.count() << " milliseconds" << std::endl;
}

void demonstrate_training_with_blocks() {
    std::cout << "\n=== Training Demonstration with Advanced Blocks ===" << std::endl;
    
    // Create a compact model for demonstration
    Model model;
    
    // MobileNet-inspired architecture
    model.add_layer<Conv2DLayer>(3, 32, 3, 3, 2, 2, 1, 1);
    model.add_layer<BatchNormLayer>(32);
    model.add_layer<ReLULayer>();
    
    // Depthwise separable block
    model.add_layer<DepthwiseConv2DLayer>(32, 3, 3, 1, 1, 1, 1);
    model.add_layer<BatchNormLayer>(32);
    model.add_layer<ReLULayer>();
    model.add_layer<Conv2DLayer>(32, 64, 1, 1, 1, 1, 0, 0);
    model.add_layer<BatchNormLayer>(64);
    model.add_layer<ReLULayer>();
    
    // Residual connection
    model.add_layer<ResidualBlock>(64, 64, 1);
    
    // Final layers
    model.add_layer<AvgPoolingLayer>(16);
    model.add_layer<FlattenLayer>();
    model.add_layer<LinearLayer>(64, 10);
    model.add_layer<SoftmaxLayer>();
    
    // Create synthetic dataset
    const int batch_size = 4;
    const int num_batches = 5;
    
    Tensor input({batch_size, 3, 32, 32});
    Tensor target({batch_size, 10});
    
    // Initialize synthetic data
    float* input_data = input.data();
    float* target_data = target.data();
    
    for (int i = 0; i < input.size(); ++i) {
        input_data[i] = (float)rand() / RAND_MAX;
    }
    
    for (int b = 0; b < batch_size; ++b) {
        int class_label = rand() % 10;
        for (int c = 0; c < 10; ++c) {
            target_data[b * 10 + c] = (c == class_label) ? 1.0f : 0.0f;
        }
    }
    
    // Create optimizer and loss
    auto optimizer = make_adam_optimizer(
        model.get_parameters(),
        model.get_gradients(),
        0.001f
    );
    
    auto loss_fn = make_crossentropy_loss();
    
    std::cout << "Training compact MobileNet + ResNet hybrid model..." << std::endl;
    std::cout << "Batch size: " << batch_size << ", Batches: " << num_batches << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < num_batches; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Forward pass
        Tensor predictions = model.forward(input, true);
        float loss = loss_fn->compute(predictions, target);
        
        // Backward pass
        optimizer->zero_grad();
        Tensor loss_grad = loss_fn->gradient(predictions, target);
        model.backward(loss_grad);
        optimizer->step();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Epoch " << epoch + 1 << "/" << num_batches 
                  << " - Loss: " << std::fixed << std::setprecision(6) << loss
                  << " - Time: " << duration.count() << "ms" << std::endl;
    }
    
    std::cout << "Training completed successfully!" << std::endl;
}

int main() {
    std::cout << "CNN Advanced Architectural Blocks Demo" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        // Seed random number generator for reproducible results
        srand(42);
        
        // Test each advanced block
        test_depthwise_convolution();
        test_residual_block();
        test_inception_module();
        test_bottleneck_block();
        
        // Demonstrate training
        demonstrate_training_with_blocks();
        
        std::cout << "\n=== All Advanced Blocks Tests Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
